import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp

def dist_fn(states, stickers, max_dist):
    return jnp.linalg.norm(states[:,jnp.newaxis,:] - stickers, axis=2) < max_dist
dist_fn = jax.jit(dist_fn)

def density_grouping(states, distance, distance_based=True):        
    def add_sticker(cur_f_s_idx, stickers, must_sticker_idxs, num_stickers, sticker_indices):
        stickers[num_stickers] = states[cur_f_s_idx]
        must_sticker_idxs.append([cur_f_s_idx])
        sticker_indices.append(cur_f_s_idx) 
        num_stickers += 1
        return stickers, must_sticker_idxs, num_stickers, sticker_indices
    
    min_dist = distance * 0.9
    max_dist = distance * 1.1
    
    sticker_indices = []
    stickers, must_sticker_idxs, num_stickers, sticker_indices = add_sticker(cur_f_s_idx=0, stickers=np.zeros_like(states), must_sticker_idxs=[], num_stickers=0, sticker_indices=sticker_indices)  
    
    for i in tqdm(range(1, len(states)), desc="Step (1): Assigning observations to stickers"):     
        stickers_dist = np.sqrt(np.square(np.array(states[i]) - stickers[:num_stickers]).sum(-1))
        min_idx = np.argmin(stickers_dist)
        min_sticker_dist = stickers_dist[min_idx]
        if min_sticker_dist > min_dist:
            stickers, must_sticker_idxs, num_stickers, sticker_indices = add_sticker(cur_f_s_idx=i, stickers=stickers, must_sticker_idxs=must_sticker_idxs, num_stickers=num_stickers, sticker_indices=sticker_indices)
        else:
            must_sticker_idxs[min_idx].append(i)
    stickers = stickers[:num_stickers]
    
    if distance_based == False:
        return np.array(stickers)
    
    near_sticker_idxs = [set() for _ in range(num_stickers)]
    mini_batch = 500
    size = len(stickers) // mini_batch
    
    stickers_dist_matrix = jnp.zeros((states.shape[0], stickers.shape[0]), dtype=bool)

    for i in tqdm(range(size + 1), desc='Processing batches'):
        batch_start = mini_batch * i
        batch_end = min(mini_batch * (i + 1), len(stickers))
        stickers_dist_matrix = stickers_dist_matrix.at[:, batch_start:batch_end].set(
            dist_fn(states, stickers[batch_start:batch_end], max_dist))

    stickers_dist_matrix = stickers_dist_matrix.T
    
        
    for sticker_idx in tqdm(range(num_stickers), desc="Step (2-1): Calculating near stickers"):  
        valid_f_s_idxs = np.where(stickers_dist_matrix[:, sticker_idx] < max_dist)[0]
        near_sticker_idxs[sticker_idx].update(valid_f_s_idxs)
    near_sticker_idxs = [sorted(list(s)) for s in near_sticker_idxs]
    for sticker_idx in tqdm(range(num_stickers), desc="Step (2-2): Merging and sorting stickers"):  
        must_idxs_set = set(must_sticker_idxs[sticker_idx])
        remaining_idxs = [idx for idx in near_sticker_idxs[sticker_idx] if idx not in must_idxs_set] 
        near_sticker_idxs[sticker_idx] = must_sticker_idxs[sticker_idx] + remaining_idxs 
    
    center = np.mean(states, axis=0)
    value = -np.sqrt(np.square(center - states).sum(-1))
    value = value + value.min() + 1e-6
    near_sticker_idxs = [np.array(near_list) for near_list in near_sticker_idxs]
    
    for i in tqdm(range(num_stickers), desc="Step (3): Refining sticker positions"):  
        near_list = near_sticker_idxs[i]
        must_to_near_ptr = set(np.arange(len(must_sticker_idxs[i])))
        prev_density = 0
        next_mean = stickers[i]
        inside_to_near_ptr = must_to_near_ptr
        while (prev_density < len(inside_to_near_ptr)) and (must_to_near_ptr.issubset(inside_to_near_ptr)):
            stickers[i] = next_mean
            prev_density = len(inside_to_near_ptr)
            selected_values = value[near_list[list(inside_to_near_ptr)]]
            selected_f_s = states[near_list[list(inside_to_near_ptr)], :]

            candidate_mean = np.average(selected_f_s, weights=selected_values, axis=0)
            distances = np.sqrt(np.square(selected_f_s - candidate_mean).sum(-1))
            closest_idx = np.argmin(distances)  
            next_mean = selected_f_s[closest_idx]
            next_dists = np.sqrt(np.square(np.array(next_mean) - states[near_list, :]).sum(-1))
            inside_to_near_ptr = set(np.where(next_dists < (max_dist/2))[0])
    return np.array(stickers)