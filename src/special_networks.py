from jaxrl_m.typing import *
from jaxrl_m.networks import *
import jax

class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x
    
class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)

class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)
    
class RelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = 'concat' 
    bottleneck: bool = True  # Meaning that we're using this representation for high-level actions

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.visual:
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)

        if self.bottleneck:
            rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)

        return rep

class MonolithicVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    obs_rep: int = 0
    ensemble: bool = True

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False, ensemble=self.ensemble)

    def __call__(self, observations, goals=None, info=False):
        phi = observations
        psi = goals
        if self.ensemble:
            v1, v2 = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)
            return v1, v2
            
        else:
            v = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)
            return v
            
class MonolithicQF(nn.Module):
    hidden_dims: tuple = (256, 256)
    use_layer_norm: bool = True
    bilinear: int = 0
    
    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        if self.bilinear:
            self.s_a = repr_class((*self.hidden_dims, self.bilinear), activate_final=False)
            self.s_g = repr_class((*self.hidden_dims, self.bilinear), activate_final=False)
            
        else:
            self.q_net = repr_class((*self.hidden_dims, 1), activate_final=False)
    def __call__(self, observations, actions, subgoals=None, goals=None, info=False):

        if self.bilinear:
            if goals is not None:
                s_a = self.s_a(jnp.concatenate([observations, actions], axis=-1))
                s_g = self.s_g(jnp.concatenate([observations, subgoals, goals], axis=-1))
                
            else:
                s_a = self.s_a(jnp.concatenate([observations, actions], axis=-1))
                s_g = self.s_g(jnp.concatenate([observations, subgoals], axis=-1))
            
            einsum_str = 'ijk,ijk->ij' if len(s_a.shape) == 3 else 'ijkl,ijkl->ijk'
            q1, q2 = jnp.einsum(einsum_str, s_a, s_g)
        
        else:
            
            if goals is not None:
                q1, q2 = self.q_net(jnp.concatenate([observations, actions, subgoals, goals], axis=-1)).squeeze(-1)
            else:
                q1, q2 = self.q_net(jnp.concatenate([observations, actions, subgoals], axis=-1)).squeeze(-1)
            

        if info:
            return {
                'q': (q1 + q2) / 2,
            }
        return q1, q2

class HILP_GoalConditionedPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256) # (512, 512, 512)
    readout_size: tuple = (256,)
    skill_dim: int = 2 # 32
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None
    obs_dim: int = 0
    detach: int = 1
    expected_value: float = 0.0
    
    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.skill_dim), activate_final=False, ensemble=self.ensemble)
        
        if self.encoder is not None: 
            phi = nn.Sequential([self.encoder(), phi])
        self.phi = phi

    def get_phi(self, observations):
        return self.phi(observations)[0]  # Use the first vf

    def __call__(self, observations, goals=None, info=False):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)

        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
        
        return v

def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)
  
class LatentSubgoalModelDecode(nn.Module):
    hidden_dim: Sequence[int]
    output_shape: Sequence[int] 
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    layer_norm : bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()
    dropout_rate : float = 0.1
    activate_final: int = False

    @nn.compact
    def __call__(self, state, z, training=False) -> jnp.ndarray:

        x = jnp.concatenate([state, z], axis=-1)
        for i, size in enumerate(self.hidden_dim):
            x = nn.Dense(size, kernel_init = self.kernel_init)(x)
            x = self.activations(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        if self.layer_norm:
            x = nn.LayerNorm()(x)

        subgoal = nn.Dense(self.output_shape, kernel_init=self.kernel_init)(x)

        return subgoal

class LatentlModel(nn.Module):
    hidden_dim: Sequence[int]
    output_shape: Sequence[int] 
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    layer_norm : bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()
    latent_dim : int = 10
    rng : PRNGKey = jax.random.PRNGKey(0)

    def setup(self) -> None:

        self.z_encoder = LayerNormMLP(self.hidden_dim, activate_final=True)
        self.z_std, self.z_mean = nn.Dense(self.latent_dim, kernel_init=self.kernel_init), nn.Dense(self.latent_dim, kernel_init=self.kernel_init)

        self.z_decoder = LatentSubgoalModelDecode(hidden_dim=self.hidden_dim, output_shape=self.output_shape)

    def __call__(self, state, targets, training=False, seed=None) -> jnp.ndarray:
        if seed is None:
            seed = self.rng

        x = jnp.concatenate([state, targets], axis=-1)
        h = self.z_encoder(x)
        z_mean = self.z_mean(h)
        z_log_std = self.z_std(h)
        z_std = jnp.exp(jnp.clip(z_log_std, -5, 2))
        z = z_mean + z_std * jax.random.normal(seed, shape=(*z_std.shape,))
        subgoal = self.z_decoder(state, z, training=training)

        return z, z_mean, z_std, subgoal

    def get_z(self, state, targets, seed=None) -> jnp.ndarray:
        if seed is None:
            seed = self.rng

        x = jnp.concatenate([state, targets], axis=-1)
        h = self.z_encoder(x)
        z_mean = self.z_mean(h)
        z_log_std = self.z_std(h)
        z_std = jnp.exp(jnp.clip(z_log_std, -5, 2))
        z = z_mean + z_std * jax.random.normal(seed, shape=(*z_std.shape,))

        return z, z_mean, z_std

    def get_decode(self, state, z, training=False):
        subgoal = self.z_decoder(state, z, training=training)
        return subgoal        
    
class PriorModel(nn.Module):
    hidden_dim: Sequence[int]
    output_shape: Sequence[int] 
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm : bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, state) -> jnp.ndarray:
        x = state
        for i, size in enumerate(self.hidden_dim):
            x = nn.Dense(size, kernel_init = self.kernel_init)(x)
            x = self.activations(x)
                
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        
        mean = nn.Dense(self.output_shape, kernel_init=self.kernel_init)(x)
        log_std = nn.Dense(self.output_shape, kernel_init=self.kernel_init)(x)
        std = jnp.exp(jnp.clip(log_std, -5, 2))
        
        return mean, std

def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)
