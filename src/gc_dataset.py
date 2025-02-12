from jaxrl_m.dataset import Dataset
from jaxrl_m.typing import *
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import ml_collections
from typing import *

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    terminal: bool = True
    use_rep: str = ""
    key_nodes: Any = None
    
    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': 0.0,
            'terminal': True,
        })

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)

        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)

        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)

        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx
    
    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)
        
        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])
        return batch

@dataclasses.dataclass
class GCSDataset(GCDataset):
    way_steps: int = 40
    high_p_randomgoal: float = 0.
    high_p_relabel : float = 0.
    env_name: str = None
    temporal_dist: float = 0
    rng: Any = jax.random.PRNGKey(0)
    
    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': 0.0,
            'terminal': False,
        })

    def sample(self, batch_size: int, indx=None, **kwargs):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)
        success = (indx == goal_indx)

        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['masks'] = (1.0 - success.astype(float))
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])
                
        if isinstance(batch['goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])

        return batch
    
    def td_sample(self, batch_size: int, indx=None, **kwargs):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)
        success = (indx == goal_indx)

        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['masks'] = (1.0 - success.astype(float))
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])
                
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
      
        if 'hilp_observations' in self.dataset.keys():
            batch['hilp_goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['hilp_observations'])
            hilp_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)
            batch['hilp_random_sg'] = jax.tree_map(lambda arr: arr[hilp_random_goal_indx], self.dataset['hilp_observations'])
            batch['hilp_final_goals'] = jax.tree_map(lambda arr: arr[final_state_indx], self.dataset['hilp_observations'])

        if kwargs.get('high_actor_update') == True or kwargs.get('prior_update') == True :
            batch['hilp_high_observations'] = batch['hilp_observations'] 
            distance = np.random.rand(batch_size)
            
            high_target_idx = self.select_bouned_target(batch_size, batch, indx, final_state_indx)
            self.rng, subkey1 = jax.random.split(self.rng)
            batch['diff_indx'] = high_target_idx - indx
            
            high_traj_goal_indx = np.round((np.minimum(high_target_idx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

            high_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)

            pick_random = (np.random.rand(batch_size) < self.high_p_randomgoal)
            high_goal_idx = np.where(pick_random, high_random_goal_indx, high_traj_goal_indx)

            high_goal_idx = np.where(np.random.rand(batch_size) < self.high_p_relabel, indx, high_goal_idx)
            
            batch['hilp_high_goals'] = jax.tree_map(lambda arr: arr[high_goal_idx], self.dataset['hilp_observations'])
            
            batch['hilp_high_targets'] = jax.tree_map(lambda arr: arr[high_target_idx], self.dataset['hilp_observations'])
                
        if isinstance(batch['goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])

        return batch
    
    def select_bouned_target(self, batch_size, batch, indx, final_state_indx):
        way_steps_multiples = (jnp.arange(1, 6) * (self.way_steps//2))[:, jnp.newaxis]  
        high_traj_target_indx_candidates = jnp.minimum(indx + way_steps_multiples, final_state_indx)
        high_traj_target_indx_candidates = high_traj_target_indx_candidates.T 

        def get_candidates(arr):
            def index_single_batch(indices):
                return arr[indices]  
            return index_single_batch(high_traj_target_indx_candidates) 

        high_traj_target_candidates = jax.tree_map(get_candidates, self.dataset['hilp_observations']).transpose(1, 0, 2)  # (n, batch_size, feature_dim)

        batch_hilp_observations = batch['hilp_observations']
        high_target_dist_from_hilp_observation = jnp.linalg.norm(batch_hilp_observations - high_traj_target_candidates, axis=-1).T

        mask = (high_target_dist_from_hilp_observation >= self.temporal_dist-5) & (high_target_dist_from_hilp_observation <= self.temporal_dist+5)
        
        key, subkey1 = jax.random.split(self.rng)
        self.rng = key
    
        random_indx = jax.random.randint(subkey1, shape=(batch_size,), minval=0, maxval=5)  # (batch_size,)

        key, subkey2 = jax.random.split(key)
        random_scores = jax.random.uniform(subkey2, shape=(batch_size, 5))  # (batch_size, n)
        masked_scores = jnp.where(mask, random_scores, -jnp.inf)  # (batch_size, 5)
        masked_indx = jnp.argmax(masked_scores, axis=1)  # (batch_size,)
        has_true = jnp.any(mask, axis=1)  # (batch_size,)
        selected_indx = jnp.where(has_true, masked_indx, random_indx)# (batch_size,)
        selected_target_indices = high_traj_target_indx_candidates[jnp.arange(batch_size), selected_indx]

        return selected_target_indices

@jax.jit
def fine_anchor_states(targets, anchor_states):
    dist = jnp.linalg.norm(targets[:,np.newaxis,:] - anchor_states, axis=-1)
    index = jnp.argmin(dist, axis=-1)
    sampled_anchor_states = jax.tree_map(lambda arr: arr[index], anchor_states)
    
    return sampled_anchor_states  