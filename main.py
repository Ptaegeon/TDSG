import os
import jax
import flax
import tqdm
import time
import wandb
import pickle
import datetime
import tensorflow as tf
import numpy as np
from absl import app, flags
from functools import partial
from src.gc_dataset import GCSDataset, fine_anchor_states
from ml_collections import config_flags
from src.utils import record_video
from jaxrl_m.wandb import setup_wandb, default_wandb_config
from src.agents import tdsg as learner
from src import d4rl_utils
from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories
from src import sticker_utils as sticker

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', f'experiment_output/', '')
flags.DEFINE_string('run_group', 'EXP', '')
flags.DEFINE_string('env_name', 'kitchen-mixed-v0', '')

flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('batch_size', 1024, '')
flags.DEFINE_integer('pretrain_steps', 500002, '')
flags.DEFINE_integer('eval_interval', 100000, '')
flags.DEFINE_integer('save_interval', 100000, '')
flags.DEFINE_integer('log_interval', 1000, '')
flags.DEFINE_integer('eval_episodes', 5, '')
flags.DEFINE_integer('num_video_episodes', 6, '')

flags.DEFINE_integer('use_layer_norm', 1, '')
flags.DEFINE_integer('value_hidden_dim', 512, '')
flags.DEFINE_integer('value_num_layers', 3, '')
flags.DEFINE_integer('actor_hidden_dim', 512, '')
flags.DEFINE_integer('actor_num_layers', 3, '')
flags.DEFINE_integer('qf_hidden_dim', 512, '')
flags.DEFINE_integer('qf_num_layers', 3, '')
flags.DEFINE_integer('geom_sample', 1, '')

flags.DEFINE_float('p_randomgoal', 0.375, '') 
flags.DEFINE_float('p_trajgoal', 0.625, '')
flags.DEFINE_float('p_currgoal', 0.0, '') 
flags.DEFINE_float('high_p_randomgoal', 0.5, '') 
flags.DEFINE_float('high_p_relabel', 0.2, '')
flags.DEFINE_float('high_temperature', 10, '')
flags.DEFINE_float('hilp_pretrain_expectile', 0.95, '')
flags.DEFINE_float('pretrain_expectile', 0.7, '')
flags.DEFINE_float('low_pretrain_expectile', 0.9, '')
flags.DEFINE_float('temperature', 10, '')
flags.DEFINE_float('discount', 0.99, '')

flags.DEFINE_integer('subgoal_dim', 10, '')
flags.DEFINE_integer('td_dim', 32, '')
flags.DEFINE_integer('anchor_interval', 20, '') 
flags.DEFINE_float('temporal_dist', 10, '') 

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'Test',
    'group': 'Debug',
    'name': '{env_name}',
})
    
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)
gcdataset_config = GCSDataset.get_default_config()
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)

def main(_):
    g_start_time = time.strftime('%m-%d_%H-%M')

    exp_name = ''
    exp_name += f'{FLAGS.wandb["name"]}'
    exp_name += f'_{g_start_time}'
    
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'

    FLAGS.gcdataset['p_randomgoal'] = FLAGS.p_randomgoal
    FLAGS.gcdataset['p_trajgoal'] = FLAGS.p_trajgoal
    FLAGS.gcdataset['p_currgoal'] = FLAGS.p_currgoal
    FLAGS.gcdataset['high_p_randomgoal'] = FLAGS.high_p_randomgoal
    FLAGS.gcdataset['high_p_relabel'] = FLAGS.high_p_relabel
    FLAGS.gcdataset['geom_sample'] = FLAGS.geom_sample
    FLAGS.gcdataset['discount'] = FLAGS.discount
    FLAGS.gcdataset['temporal_dist'] = FLAGS.temporal_dist
    FLAGS.gcdataset['env_name'] = FLAGS.env_name
  
    FLAGS.config['env_name'] = FLAGS.env_name
    FLAGS.config['pretrain_expectile'] = FLAGS.pretrain_expectile
    FLAGS.config['high_temperature'] = FLAGS.high_temperature
    FLAGS.config['temperature'] = FLAGS.temperature
    FLAGS.config['discount'] = FLAGS.discount
    FLAGS.config['value_hidden_dims'] = (FLAGS.value_hidden_dim,) * FLAGS.value_num_layers
    FLAGS.config['actor_hidden_dims'] = (FLAGS.actor_hidden_dim,) * FLAGS.actor_num_layers
    FLAGS.config['qf_hidden_dims'] = (FLAGS.qf_hidden_dim,) * FLAGS.qf_num_layers
    FLAGS.config['td_dim'] = FLAGS.td_dim 
    
    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    FLAGS.wandb['name'] = FLAGS.wandb['exp_descriptor'] = exp_name   
    FLAGS.wandb['group'] = FLAGS.wandb['exp_prefix'] = FLAGS.run_group 

    setup_wandb(params_dict, **FLAGS.wandb)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb['group'], FLAGS.env_name + "_" + FLAGS.wandb['name'].split('_', 2)[2])
    os.makedirs(FLAGS.save_dir, exist_ok=True)
        
    log_file_name = f"log_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.txt"
    log_file_path = os.path.join(FLAGS.save_dir, log_file_name)
    with open(log_file_path, 'w') as log_file:
        log_file.write("Flags used:\n")
        for flag_name in FLAGS:
            flag_value = getattr(FLAGS, flag_name)
            log_file.write(f"{flag_name}: {flag_value}\n")
    print(f"Log file created at {log_file_path}")

    
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    env, dataset, dataset_config = d4rl_utils.make_env_get_dataset(FLAGS)
    total_steps = FLAGS.pretrain_steps
    example_observation = dataset['observations'][0, np.newaxis]
    example_action = dataset['actions'][0, np.newaxis]

    FLAGS.config.update(dataset_config)
    
    agent = learner.create_learner(FLAGS.seed,
                                   example_observation,
                                   example_action,
                                   flag=FLAGS,
                                   **FLAGS.config)
    
    pretrain_dataset = GCSDataset(dataset, **FLAGS.gcdataset.to_dict())

    base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
    env.reset()
    
    pretrain_steps = 0
    train_step = 2*10**5 +1
    td_train_steps = train_step
    update = dict(hilp_update=True)
    for i in tqdm.tqdm(range(1, td_train_steps),
                desc="TD_train",
                smoothing=0.1,
                dynamic_ncols=True):
        
        pretrain_batch = pretrain_dataset.sample(FLAGS.batch_size, **update)
        agent, update_info = agent.pretrain_update(pretrain_batch, **update)
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
    
    
    # Anchor state selection 
    hilp_observations = d4rl_utils.get_hilp_obs(agent, dataset['observations'], FLAGS)
    hilp_next_observations = d4rl_utils.get_hilp_obs(agent, dataset['next_observations'], FLAGS)
    anchor_states = sticker.density_grouping(hilp_observations, FLAGS.anchor_interval)
   
    
    # subgoal model training
    pretrain_steps = td_train_steps
    subgoal_model_train_steps = train_step
    update = dict(prior_update=True)
    dataset = dataset.copy({'hilp_observations':hilp_observations, 'hilp_next_observations':hilp_next_observations})
    pretrain_dataset = GCSDataset(dataset, **FLAGS.gcdataset.to_dict())

    for i in tqdm.tqdm(range(1, subgoal_model_train_steps),
                desc="subgoal_model_train",
                smoothing=0.1,
                dynamic_ncols=True):
        pretrain_batch = pretrain_dataset.td_sample(FLAGS.batch_size , **update)
        agent, update_info = agent.pretrain_update(pretrain_batch, **update)
        
        if i % FLAGS.log_interval == 0:     
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=pretrain_steps + i)
            
    pretrain_steps += subgoal_model_train_steps
    total_steps = FLAGS.pretrain_steps
    update = dict(actor_update=True, high_actor_update=True)
        
    FLAGS.gcdataset['p_randomgoal'] = 0.3
    FLAGS.gcdataset['p_trajgoal'] = 0.5
    FLAGS.gcdataset['p_currgoal'] = 0.2
    pretrain_dataset = GCSDataset(dataset, **FLAGS.gcdataset.to_dict())
                                      
    for i in tqdm.tqdm(range(1, total_steps + 1),
                   desc="main_train",
                   smoothing=0.1,
                   dynamic_ncols=True):
        
        pretrain_batch = pretrain_dataset.td_sample(FLAGS.batch_size, **update)
        pretrain_batch['anchor_states'] = fine_anchor_states(targets=pretrain_batch['hilp_high_targets'], anchor_states=anchor_states)
            
        agent, update_info = agent.pretrain_update(pretrain_batch, **update)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=pretrain_steps+i)

        if i == 1 or i % FLAGS.eval_interval == 0:

            eval_episodes = 1 if i == 1 else FLAGS.eval_episodes
            num_video_episodes = 1 if i == 1 else FLAGS.num_video_episodes
            
            policy_fn = partial(supply_rng(agent.sample_actions))
            high_policy_fn = partial(supply_rng(agent.sample_high_actions))
            base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])

            eval_info, renders = evaluate_with_trajectories(
                    policy_fn=policy_fn, high_policy_fn=high_policy_fn, env=env,
                    env_name=FLAGS.env_name, num_episodes=eval_episodes,
                    base_observation=base_observation, num_video_episodes=num_video_episodes,
                    eval_temperature=0,
                    config=FLAGS.config,
                    FLAGS=FLAGS,
                    agent=agent)
            
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            if FLAGS.num_video_episodes > 0 and len(renders):
                video = record_video('Video', i, renders=renders)
                eval_metrics['video'] = video
                
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=FLAGS.config.to_dict(),
                FLAGS={name: FLAGS[name].value for name in FLAGS}
            )
            if i == 1 or i % FLAGS.save_interval == 0:                
                fname = os.path.join(FLAGS.save_dir, f'params_{i}.pkl')
                print(f'Saving to {fname}')
                with open(fname, "wb") as f:
                    pickle.dump(save_dict, f)         
                
            wandb.log(eval_metrics, step=pretrain_steps+i)

if __name__ == '__main__':
    import random
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    
    app.run(main)
