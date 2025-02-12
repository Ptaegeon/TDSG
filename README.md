# TDSG: Temporal Distance-aware Subgoal Generation for Offline Hierarchical Reinforcement Learning

## Installation

```
conda create --name ask python=3.8
conda activate tdsg
pip install -r requirements.txt --no-deps
pip install "jax[cuda11_cudnn82]==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
## Script

```
# antmaze-ultra-diverse
python main.py --save_dir "experiment_output" --run_group "ant-ultra-diverse"  --seed 0 --env_name "antmaze-ultra-diverse-v0" --pretrain_steps 500002 --eval_interval 100000 --save_interval 100000 --subgoal_dim 10 --td_dim 32 temporal_dist 15 --anchor_interval 20 --hilp_pretrain_expectile 0.9 --pretrain_expectile 0.7 --low_pretrain_expectile 0.9 --high_temperature 5 --temperature 10

# antmaze-large-diverse
python main.py --save_dir "experiment_output" --run_group "ant-large-diverse" --seed 0 --env_name "antmaze-large-diverse-v2" --pretrain_steps 500002 --eval_interval 100000 --save_interval 100000 --subgoal_dim 10 --td_dim 32 temporal_dist 15 --anchor_interval 20 --hilp_pretrain_expectile 0.9 --pretrain_expectile 0.7 --low_pretrain_expectile 0.9 --high_temperature 5 --temperature 10

# kitchen-mixed
python main.py --save_dir "experiment_output" --run_group "kitchen-mixed" --seed 0 --env_name kitchen-mixed-v0 --pretrain_steps 500002 --eval_interval 25000 --save_interval 25000 --subgoal_dim 10 --td_dim 32 temporal_dist 40 --anchor_interval 20 --hilp_pretrain_expectile 0.95 --pretrain_expectile 0.7 --low_pretrain_expectile 0.7 --high_temperature 5 --temperature 10


## License
MIT
