from typing import Dict
import jax
import gym
import numpy as np
from collections import defaultdict
import time
import tqdm
import jax.numpy as jnp

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped

def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img
        
def evaluate_with_trajectories(
        policy_fn, high_policy_fn, env: gym.Env, env_name, num_episodes: int, base_observation=None, num_video_episodes=0,
        eval_temperature=0, 
        config=None, FLAGS=None, agent=None, **kwargs
) -> Dict[str, float]:
    stats = defaultdict(list)
    renders = []
        
    for i in tqdm.tqdm(range(num_episodes + num_video_episodes)):
        trajectory = defaultdict(list)
        observation, done = env.reset(), False
        # Set goal
        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
            obs_goal = np.array(obs_goal)
        elif 'kitchen' in env_name:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
        else:
            raise NotImplementedError
        
        render = []
        step = 0
            
        hilp_fn = agent.get_hilp_phi
        decode = jax.jit(agent.get_decode, static_argnames=('deterministic',))
        
        hilp_observation = hilp_fn(observations=observation)
        hilp_obs_goal = hilp_fn(observations=obs_goal)
            
        while not done:                
            hilp_observation = hilp_fn(observations=observation)
            cur_obs_subgoal = high_policy_fn(observations=hilp_observation, goals=hilp_obs_goal, temperature=eval_temperature)

            prior_input = jnp.concatenate([hilp_observation, hilp_obs_goal], axis=-1)
            recon_subgoal_to_hilp = decode(observations=prior_input, z=cur_obs_subgoal, deterministic=True)
        
            skill = recon_subgoal_to_hilp - hilp_observation
            skill = skill / jnp.linalg.norm(skill)
            action = policy_fn(observations=observation, goals=skill, temperature=eval_temperature)
            
            if 'antmaze' in env_name:
                next_observation, r, done, info = env.step(action)
            elif 'kitchen' in env_name:
                next_observation, r, done, info = env.step(action)
                next_observation = next_observation[:30]
   
            step += 1
 
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
            
            # Render
            if i >= num_episodes and (step % 3 == 0 or 'kitchen' in env_name):

                if 'antmaze' in env_name:
                    size = 240
                    cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                    render.append(cur_frame)
                    
                elif 'kitchen' in env_name:
                    cur_frame = kitchen_render(env, wh=200).transpose(2, 0, 1)
                    render.append(cur_frame)
     
        add_to(stats, flatten(info, parent_key="final"))
        if i >= num_episodes:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    
    return stats, renders

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )
                
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self.total_timesteps = 0
        self._reset_stats()
        return self.env.reset()
