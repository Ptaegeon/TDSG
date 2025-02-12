import gym
import d4rl
import numpy as np
from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import EpisodeMonitor

def make_env_get_dataset(FLAGS):
    env_name = FLAGS.env_name
    dataset_config = {}
    
    if 'antmaze' in FLAGS.env_name:
        if FLAGS.env_name.startswith('antmaze'):
            env_name = FLAGS.env_name
        else:
            env_name = '-'.join(FLAGS.env_name.split('-')[1:])
            
        if 'ultra' in FLAGS.env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
            env = EpisodeMonitor(env)
            
        else:
            env = make_env(env_name)
        env.seed(FLAGS.seed)
    
        dataset, dataset_config = get_dataset(env, FLAGS.env_name, flag=FLAGS)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})
        
        env.render(mode='rgb_array', width=500, height=500)
        if 'large' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
            
        elif 'ultra' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
                
    elif 'kitchen' in FLAGS.env_name:
        env = make_env(FLAGS.env_name)
        env.seed(FLAGS.seed)
        dataset, dataset_config = get_dataset(env, FLAGS.env_name, filter_terminals=True, flag=FLAGS)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
    else:
        raise NotImplementedError
    
    return env, dataset, dataset_config

def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

def get_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                obs_dtype=np.float32,
                flag=None
                ):
        goal_info = None
        config = dict()
        if dataset is None:
            dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dataset['terminals'][-1] = 1
        if filter_terminals: 
            non_last_idx = np.nonzero(~dataset['terminals'])[0]
            last_idx = np.nonzero(dataset['terminals'])[0]
            penult_idx = last_idx - 1
            new_dataset = dict()
            if 'visual' in flag.env_name:
                goal_info = dataset['goal_info']
                del(dataset['goal_info'])
            
            for k, v in dataset.items():
                if k == 'terminals':
                    v[penult_idx] = 1
                new_dataset[k] = v[non_last_idx]
            dataset = new_dataset

        if 'antmaze' in env_name:
            # antmaze: terminals are incorrect for GCRL
            dones_float = np.zeros_like(dataset['rewards'])
            dataset['terminals'][:] = 0.
            for i in range(len(dones_float) - 1):
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            dones_float[-1] = 1
           
        elif 'kitchen' in env_name:
            dones_float = dataset['terminals'].copy()
        
        else:
            NotImplementedError
        
        observations = dataset['observations'].astype(obs_dtype)
        next_observations = dataset['next_observations'].astype(obs_dtype)

        if 'kitchen' in flag.env_name:
            config = {'observation_min':dataset['observations'][:,:30].min(axis=0),
                'observation_max':dataset['observations'][:,:30].max(axis=0),
                }
        else:
            config = {'observation_min':dataset['observations'].min(axis=0),
                'observation_max':dataset['observations'].max(axis=0),
                }
            
        return Dataset.create(
            observations=observations,
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=next_observations,
            ), config

def get_hilp_obs(agent, observations, flags):
    rep_obs = np.zeros((len(observations), flags.td_dim), dtype=np.float32)
    encoder_fn = agent.get_hilp_phi
    mini_batch = 20000
    size = len(observations) // mini_batch
    for i in range(size+1):
        rep_obs[mini_batch*i:mini_batch*(i+1)] = encoder_fn(observations=observations[mini_batch*i:mini_batch*(i+1)])
    
    return rep_obs

def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img
