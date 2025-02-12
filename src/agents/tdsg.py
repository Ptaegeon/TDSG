from jaxrl_m.typing import *
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy
import flax
import ml_collections
from src.special_networks import MonolithicQF,  HILP_GoalConditionedPhiValue, PriorModel, MonolithicVF, LatentlModel

def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

def compute_kl(post_mean, post_std, prior_mean, prior_std=1):
    kl = jnp.log(prior_std) - jnp.log(post_std) + 0.5 * ((post_std**2 + (post_mean - prior_mean)**2) / prior_std**2 - 1)
    return kl

def compute_actor_loss(agent, batch, network_params):           
    z = (batch['hilp_next_observations'] - batch['hilp_observations']) 

    epsilon = 1e-10
    high_actions_unnormalized_skills  = batch['high_actions'] - batch['hilp_observations']
    norm = jnp.linalg.norm(high_actions_unnormalized_skills, axis=1, keepdims=True) + epsilon
    high_actions_normalized_skills = high_actions_unnormalized_skills  / norm 
    batch['high_skills'] = high_actions_normalized_skills
                
    random_sg_unnormalized_skills  = batch['hilp_random_sg'] - batch['hilp_observations']
    norm = jnp.linalg.norm(random_sg_unnormalized_skills, axis=1, keepdims=True) + epsilon
    batch['dataset_goal_skills'] = random_sg_unnormalized_skills  / norm 
    batch['dataset_hilp_rewards'] = (z * batch['dataset_goal_skills']).sum(axis=1)
        
    observations = batch['observations']
    next_observations = batch['next_observations']
    subgoals = batch['high_skills']

    dist = agent.actor.apply_fn({'params':network_params}, jnp.concatenate([observations, subgoals],axis=-1))
        
    batch['q_observations'] = observations
    batch['q_next_observations'] = next_observations
    
    v = agent.low_value(observations, subgoals)
    q1, q2 = agent.qf(observations, batch['actions'], subgoals)
    q = jnp.minimum(q1, q2)
    
    adv = q - v
    exp_a = jnp.exp(adv * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)
    
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()
    

    return actor_loss, {
        'actor_loss': actor_loss,
        'exp_a' : exp_a.mean(),
        'q' : q.mean(),
        'v' : v.mean(),
        'adv' : adv.mean(),
        'mse': jnp.mean((dist.mode() - batch['actions'])**2),
        'bc_log_probs': log_probs.mean(),
        'low_scale': dist.scale_diag.mean(),
    }

def compute_high_actor_loss(agent, batch, network_params):

    cur_goals = batch['hilp_high_goals']
    observations = batch['hilp_high_observations']
    high_targets = batch['hilp_high_targets']
    
    dist = agent.high_actor.apply_fn({'params':network_params}, jnp.concatenate([batch['hilp_high_observations'], batch['hilp_high_goals']], axis=-1))
           
    v1, v2 = agent.value(observations, cur_goals)
    nv1, nv2 = agent.value(high_targets, cur_goals)
    
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    adv = nv - v
    
    exp_a = jnp.exp(adv * agent.config['high_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)
        
    prior_input = jnp.concatenate([batch['hilp_observations'], batch['hilp_high_goals']], axis=-1)
    
    _, target_z, _, _ = agent.latent(prior_input, batch['anchor_states'], seed=agent.rng)
    log_probs = dist.log_prob(target_z)
    actor_loss = -exp_a * log_probs

    temporal_dist = jnp.linalg.norm(batch['hilp_high_targets'] - batch['hilp_observations'], axis=-1)
    mask = (temporal_dist >= agent.config['temporal_dist'] - 5) & (temporal_dist <= agent.config['temporal_dist']+ 5).astype(bool)
        
    denominator = jnp.maximum(mask.sum(), 1.0)
    dim = agent.config['subgoal_dim']
    actor_loss = (actor_loss.sum(-1) * mask).sum() / (dim * denominator)

    recon_high_action = agent.latent(state=prior_input, z=dist.loc, method='get_decode')
    batch['high_actions'] = jax.lax.stop_gradient(recon_high_action)                
    
    return actor_loss, {
        'high_actor_loss': actor_loss,
        'high_adv': adv.mean(),
        'high_exp_a' : exp_a.mean(),
        'q' : nv.mean(),
        'v' : v.mean(),
        'log_probs' : log_probs.mean(),
        'high_mse': jnp.mean((dist.mode() - target_z)**2),
        'high_std': dist.scale_diag.mean(),
    }

def compute_value_loss(agent, batch, network_params):
    masks = 1.0 - batch['rewards']
    rewards = batch['rewards'] - 1.0
    
    cur_goals = batch['hilp_goals']
    observations = batch['hilp_observations']
    next_observations = batch['hilp_next_observations']

    (next_v1, next_v2) = agent.target_value(next_observations, cur_goals)
    next_v = jnp.minimum(next_v1, next_v2)
    q = rewards + agent.config['discount'] * masks * next_v

    (v1_t, v2_t) = agent.target_value(observations, cur_goals)
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = rewards + agent.config['discount'] * masks * next_v1
    q2 = rewards + agent.config['discount'] * masks * next_v2
    (v1, v2) = agent.value.apply_fn({'params':network_params}, observations, cur_goals)

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['pretrain_expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['pretrain_expectile']).mean()
    value_loss = value_loss1 + value_loss2

    return value_loss, {
        'value_loss': value_loss,
        'v max': v1.max(),
        'v min': v1.min(),
        'v mean': v1.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
    }
    
def hilp_compute_value_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    masks = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    rewards = batch['rewards'] - 1.0
    next_v1, next_v2 = agent.hilp_target_value(batch['next_observations'], batch['goals'])
    next_v = jnp.minimum(next_v1, next_v2)
        
    q = rewards + agent.config['discount'] * masks * next_v

    v1_t, v2_t = agent.hilp_target_value(batch['observations'], batch['goals'])
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = rewards + agent.config['discount'] * masks * next_v1
    q2 = rewards + agent.config['discount'] * masks * next_v2
    v1, v2= agent.hilp_value.apply_fn({'params':network_params}, batch['observations'], batch['goals'])
    v = (v1 + v2) / 2

    mask = jnp.ones(batch['observations'].shape[0]).astype(jnp.float32)
    denominator = batch['observations'].shape[0]
        
    value_loss1 = (expectile_loss(adv, q1 - v1,  agent.config['hilp_pretrain_expectile']) * mask).sum() / denominator
    value_loss2 = (expectile_loss(adv, q2 - v2,  agent.config['hilp_pretrain_expectile']) * mask).sum() / denominator
    
    value_loss = value_loss1 + value_loss2 
    
    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
        'value_loss': value_loss1 + value_loss2,
    }
    
def compute_qf_loss(agent, batch, network_params):
    observations = batch['q_observations']
    next_observations = batch['q_next_observations']
    subgoals = batch['dataset_goal_skills'] 
    rewards = batch['dataset_hilp_rewards']

    next_v = agent.low_value(next_observations, subgoals)
    q = rewards + agent.config['discount'] * next_v 
    q1, q2 = agent.qf.apply_fn({'params':network_params}, observations, batch['actions'], subgoals)
    q_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

    return q_loss, {
        'q_loss': q_loss,
        'q max': q1.max(),
        'q min': q1.min(),
        'q mean': q1.mean(),
        'q adv': (next_v - q1).mean(),
    }
    
def compute_low_value_loss(agent, batch, network_params):
    
    observations = batch['q_observations']
    subgoals = batch['dataset_goal_skills'] 
    
    (q1, q2) = agent.target_qf(observations, batch['actions'], subgoals)
    q = jnp.minimum(q1, q2)
    v = agent.low_value.apply_fn({'params':network_params}, observations, subgoals)
    adv = q - v
    value_loss = expectile_loss(adv, q - v, agent.config['low_pretrain_expectile']).mean()
    
    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
    }
    
class JointTrainAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    hilp_value: TrainState
    hilp_target_value: TrainState
    value: TrainState
    target_value: TrainState
    qf: TrainState
    target_qf: TrainState
    low_value: TrainState
    low_target_value: TrainState
    actor: TrainState
    high_actor: TrainState
    latent : TrainState
    prior : TrainState
    config: dict = flax.struct.field(pytree_node=False)
    
    def pretrain_update(agent, pretrain_batch, value_update=False, qf_update=False, low_value_update=False, actor_update=False,  high_actor_update=False, hilp_update=False, prior_update=False):

        def compute_prior_loss(params, batch):
            observations = batch['hilp_observations']
            sub_goals = batch['hilp_high_targets']

            prior_input = jnp.concatenate([observations, batch['hilp_final_goals']], axis=-1)
            _, z_mean, z_std, recon = agent.latent.apply_fn({'params': params['latent']}, prior_input, sub_goals, seed=agent.rng)
            prior_mean, prior_std = agent.prior.apply_fn({'params': params['prior']}, prior_input)
                
            recon_target = batch['hilp_high_targets']
            temporal_dist = jnp.linalg.norm(sub_goals - observations, axis=-1)
            mask = (temporal_dist >= agent.config['temporal_dist'] - 5) & (temporal_dist <= agent.config['temporal_dist'] + 5).astype(bool)
                
            denominator = jnp.maximum(mask.sum(), 1.0)
            recon_loss = (((recon - recon_target)**2).sum(-1) * mask).sum() / (agent.config['td_dim'] * denominator)

            regul_loss = compute_kl(z_mean, z_std, jax.lax.stop_gradient(prior_mean), jax.lax.stop_gradient(prior_std))
            prior_loss_ = compute_kl(jax.lax.stop_gradient(z_mean), jax.lax.stop_gradient(z_std), prior_mean, prior_std)
            subgoal_dim = agent.config['subgoal_dim']
            regul_loss = (regul_loss.sum(-1) * mask).sum() / (subgoal_dim * denominator)
            
            elbo_loss = recon_loss + agent.config['beta'] * (1 - agent.config['kl_balance']) * regul_loss
            prior_loss = agent.config['beta'] * agent.config['kl_balance'] * ((prior_loss_.sum(-1) * mask).sum() / (subgoal_dim * denominator))
                
            loss = elbo_loss + prior_loss
            
            return loss, {
                'elbo_loss': elbo_loss,
                'prior_loss': prior_loss,
                'regul_loss': regul_loss,
                'recon_loss': recon_loss,
            }
            
        def hilp_value_loss_fn(network_params):
            info = {}

            # HILP Representation
            hilp_value_loss, hilp_value_info = hilp_compute_value_loss(agent, pretrain_batch, network_params)
            for k, v in hilp_value_info.items():
                info[f'hilp_value/{k}'] = v

            return hilp_value_loss, info    
                
        def value_loss_fn(network_params):
            info = {}
            
            value_loss, value_info = compute_value_loss(agent, pretrain_batch, network_params)
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        
            return value_loss, info    
        
        def high_actor_loss_fn(network_params):
            info = {}
               
            # High Actor
            high_actor_loss, high_actor_info = compute_high_actor_loss(agent, pretrain_batch, network_params)
            for k, v in high_actor_info.items():
                info[f'high_actor/{k}'] = v

            return high_actor_loss, info    
        
        def qf_loss_fn(network_params):
            info = {}
               
            # Q function
            qf_loss, qf_info = compute_qf_loss(agent, pretrain_batch, network_params)
            for k, v in qf_info.items():
                info[f'qf/{k}'] = v

            return qf_loss, info   

        def low_value_loss_fn(network_params):
            info = {}
               
            # Q function
            low_value_loss, low_value_info = compute_low_value_loss(agent, pretrain_batch, network_params)
            for k, v in low_value_info.items():
                info[f'low_value/{k}'] = v

            return low_value_loss, info   
        
        def actor_loss_fn(network_params):
            info = {}
               
            # Actor
            actor_loss, actor_info = compute_actor_loss(agent, pretrain_batch, network_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            return actor_loss, info
             
        info = {}
        new_prior = agent.prior
        new_latent = agent.latent
        new_hilp_value = agent.hilp_value
        new_hilp_target_value = agent.hilp_target_value
        new_value = agent.value
        new_target_value = agent.target_value
        new_qf = agent.qf
        new_target_qf = agent.target_qf
        new_low_value = agent.low_value
        new_low_target_value = agent.low_target_value
        new_high_actor = agent.high_actor 
        new_actor = agent.actor 
        
        if prior_update:
            grad_fn = jax.value_and_grad(compute_prior_loss, has_aux=True)
            (_, prior_info), grads  = grad_fn({'latent':agent.latent.params, 'prior': agent.prior.params}, pretrain_batch)
            
            new_prior = agent.prior.apply_gradients(grads=grads['prior'])
            new_latent = agent.latent.apply_gradients(grads=grads['latent'])

            for k, v in prior_info.items():
                info[f'prior/{k}'] = v
            
        if hilp_update:
            new_hilp_value, hilp_value_info = agent.hilp_value.apply_loss_fn(loss_fn=hilp_value_loss_fn, has_aux=True)
            info.update(hilp_value_info)
            new_hilp_target_value = target_update(new_hilp_value, agent.hilp_target_value, agent.config['target_update_rate'])

                                
        if high_actor_update:
            new_high_actor, high_actor_info = agent.high_actor.apply_loss_fn(loss_fn=high_actor_loss_fn, has_aux=True)
            info.update(high_actor_info)
            
            new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
            info.update(value_info)
            new_target_value = target_update(new_value, agent.target_value, agent.config['target_update_rate'])
            
        if actor_update:
            new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)
            info.update(actor_info)
            
            new_qf, qf_info = agent.qf.apply_loss_fn(loss_fn=qf_loss_fn, has_aux=True)
            info.update(qf_info)
            new_target_qf = target_update(new_qf, agent.target_qf, agent.config['target_update_rate'])

            new_low_value, low_value_info = agent.low_value.apply_loss_fn(loss_fn=low_value_loss_fn, has_aux=True)
            info.update(low_value_info)
            new_low_target_value = target_update(new_low_value, agent.low_target_value, agent.config['target_update_rate'])
            
        rng, _ = jax.random.split(agent.rng, 2)
        
        return agent.replace(rng=rng, hilp_value=new_hilp_value, hilp_target_value=new_hilp_target_value, value=new_value, target_value=new_target_value,  high_actor=new_high_actor, qf=new_qf, target_qf=new_target_qf, low_value=new_low_value, low_target_value=new_low_target_value, actor=new_actor, prior=new_prior, latent=new_latent), info
    
    pretrain_update = jax.jit(pretrain_update, static_argnames=('hilp_update', 'value_update', 'qf_update', 'low_value_update', 'actor_update', 'high_actor_update', 'prior_update'))

    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       *,
                       subgoals: np.ndarray = None,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None,
                       visual: int = 0) -> jnp.ndarray:
                
        if subgoals is not None:
            dist = agent.actor(jnp.concatenate([observations, subgoals, goals], axis=-1), temperature=temperature)
        else:
            dist = agent.actor(jnp.concatenate([observations, goals], axis=-1), temperature=temperature)
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            actions = jnp.clip(actions, -1, 1)
        return actions
    sample_actions = jax.jit(sample_actions, static_argnames=('num_samples', 'low_dim_goals', 'discrete'))

    def sample_high_actions(agent,
                            observations: np.ndarray,
                            goals: np.ndarray,
                            *,
                            seed: PRNGKey,
                            temperature: float = 1.0,
                            num_samples: int = None,
                            visual: int = 0) -> jnp.ndarray:

        dist = agent.high_actor(jnp.concatenate([observations, goals], axis=-1), temperature=temperature)
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        return actions
    sample_high_actions = jax.jit(sample_high_actions, static_argnames=('num_samples',))

    # HILP 
    @jax.jit
    def get_hilp_phi(agent,
                            *,
                            observations: jnp.ndarray) -> jnp.ndarray:
        return agent.hilp_value(observations=observations, method='get_phi')

    @jax.jit
    def get_hilp_value(agent,
                            *,
                            observations: jnp.ndarray,
                            goals: jnp.ndarray) -> jnp.ndarray:

        return agent.network(observations=observations, goals=goals, method='hilp_value')
    
    def get_decode(agent,
                            *,
                            observations: jnp.ndarray,
                            z: jnp.ndarray,
                            deterministic: bool=False) -> jnp.ndarray:
        return agent.latent(state=observations, z=z, training=not deterministic, method='get_decode')

    @jax.jit
    def get_latent(agent,
                            *,
                            observations: jnp.ndarray,
                            targets: jnp.ndarray,
                            seed: bool=False) -> jnp.ndarray:
        return agent.latent(state=observations, targets=targets, seed=seed)

    @jax.jit
    def get_prior(agent,
                            *,
                            observations: jnp.ndarray) -> jnp.ndarray:
        return agent.prior(state=observations)
    
    @jax.jit
    def get_policy_rep(agent,
                       *,
                       targets: np.ndarray,
                       bases: np.ndarray = None,
                       ) -> jnp.ndarray:
        return agent.network(targets=targets, bases=bases, method='policy_goal_encoder')


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        qf_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        tau: float = 0.005,
        flag: Any = None,
        policy_lr = 1e-4,
        qf_lr = 3e-4,
        optimizer_type = 'adam',
        soft_target_update_rate = 5e-3,
        beta = 0.1,
        kl_balance=0.8,      
        **kwargs):
        
        value_def = MonolithicVF(hidden_dims=value_hidden_dims)
        low_value_def = MonolithicVF(hidden_dims=value_hidden_dims, ensemble=False)
        qf_def = MonolithicQF(hidden_dims=qf_hidden_dims)
        
        action_dim = actions.shape[-1]
        actor_def = Policy(actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        td_dim = flag.td_dim
        subgoal_dim = flag.subgoal_dim
        goal = np.zeros((1,td_dim))
        subgoal = np.zeros((1,subgoal_dim))
        high_action_dim = subgoal_dim 
        
        high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)
        
        hilp_value_goal_encoder = HILP_GoalConditionedPhiValue(hidden_dims=value_hidden_dims, ensemble=True, skill_dim=flag.td_dim)
        
        latent_def = LatentlModel(hidden_dim=actor_hidden_dims, latent_dim=subgoal_dim, output_shape=td_dim)
        prior_def = PriorModel(hidden_dim=actor_hidden_dims, output_shape=subgoal_dim)
        
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, high_actor_key, value_key, qf_key, hilp_key, prior_key = jax.random.split(rng, 7)
        
        high_actor = TrainState.create(high_actor_def, 
                                       high_actor_def.init(high_actor_key, jnp.concatenate([goal, goal], axis=-1))['params'],
                                       tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)))
        
        value = TrainState.create(value_def,
                               value_def.init(value_key, goal, goal)['params'],
                               tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)))
        target_value = TrainState.create(value_def,
                               value_def.init(value_key, goal, goal)['params'])
                
        hilp_value = TrainState.create(hilp_value_goal_encoder, 
                                       hilp_value_goal_encoder.init(hilp_key, observations, observations)['params'],
                                       tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)))
        
        hilp_target_value = TrainState.create(hilp_value_goal_encoder,
                                           hilp_value_goal_encoder.init(hilp_key, observations, observations)['params'])
        
        prior_key, dropout_key = jax.random.split(prior_key)
        rngs = {'params': prior_key, 'dropout': dropout_key, 'latent': prior_key}
        
        prior_input = jnp.concatenate([goal, goal], axis=-1)
        latent = TrainState.create(latent_def,
                            latent_def.init(rngs, prior_input, goal)['params'],
                            tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)))

        prior = TrainState.create(prior_def,
                            prior_def.init(prior_key, prior_input)['params'],
                            tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)))
        
        obs = observations
        subgoals = subgoal
        subgoals = goal
        
        actor = TrainState.create(actor_def,
                        actor_def.init(actor_key, jnp.concatenate([obs, subgoals], axis=-1))['params'],
                        tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)))
        
        qf = TrainState.create(qf_def,
                            qf_def.init(qf_key, obs, actions, subgoals)['params'],
                            tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)))
        target_qf = TrainState.create(qf_def,
                            qf_def.init(qf_key, obs, actions, subgoals)['params'])
        
        low_value = TrainState.create(low_value_def,
                            low_value_def.init(qf_key, obs, subgoals)['params'],
                            tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)))
        low_target_value = TrainState.create(low_value_def,
                            low_value_def.init(qf_key, obs, subgoals)['params'])       
        
        flag_dict = flag.flag_values_dict()
        flag_dict.update(kwargs)
        
        config = flax.core.FrozenDict(**flag_dict,  **{'target_update_rate':tau, 'policy_lr' : policy_lr, 'qf_lr' : qf_lr, 'optimizer_type' : optimizer_type, 'soft_target_update_rate' : soft_target_update_rate, 'action_dim':action_dim, 'high_action_dim':high_action_dim, 'beta':beta,
        'kl_balance':kl_balance})

        return JointTrainAgent(rng, prior=prior, latent=latent, value=value, target_value=target_value, high_actor=high_actor, qf=qf, target_qf=target_qf, low_value=low_value, low_target_value=low_target_value, actor=actor, hilp_value=hilp_value, hilp_target_value=hilp_target_value, config=config)


def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (256, 256),
        'qf_hidden_dims': (256, 256),
        'discount': 0.99,
        'temperature': 1.0,
        'tau': 0.005,
        'pretrain_expectile': 0.7,
    })

    return config