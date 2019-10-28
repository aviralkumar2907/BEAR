This is the code for NeurIPS 2019 paper Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction.

Our code is built off of the BCQ[https://github.com/sfujim/BCQ] repository. To run BEAR, please use a command like this:

```
python main.py --buffer_name=buffer_walker_300_curr_action.pkl --eval_freq=1000 --algo_name=BEAR
--env_name=Walker2d-v2 --log_dir=data_walker_BEAR/ --lagrange_thresh=10.0 
--distance_type=MMD --mode=auto --num_samples_match=5 --lamda=0.0 --version=0 
--mmd_sigma=20.0 --kernel_type=gaussian
```

```
python main.py --buffer_name=buffer_hopper_300_curr_action.pkl --eval_freq=1000 --algo_name=BEAR
--env_name=Hopper-v2 --log_dir=data_hopper_BEAR/ --lagrange_thresh=10.0 --distance_type=MMD
--mode=auto --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=10.0 --kernel_type=laplacian
```

Algorithms Supported:
1. BCQ (algo_name=BCQ) [Fujimoto et.al. ICML 2019]
2. TD3 (algo_name=TD3) [Fujimoto et.al. ICML 2018]
3. Behavior Cloning (algo_name=BC)
4. KL Control (algo_name=KLControl) [Jacques et.al. arxiv 2019]
5. Deep Q-learning from Demonstrations (algo_name=DQfD) [Hester et.al. 2017]

Hyperparameters that generally work well (for BEAR):
1. mmd_sigma=10.0, kernel_type=laplacian, num_samples_match=5, version=0 or 2, lagrange_thresh=10.0, mode=auto
2. mmd_sigmma=20.0, kernel_type=gaussian, num_samples_match=5, version=0 or 2, lagrange_thresh=10.0, mode=auto

We have removed ensembles from this version, and we just use a minimum/average over 2 Q-functions, without an ensemble-based conservative estimate based on sample variance. This is because we didn't find ensemble variance to in general provide benefits, although it doesn't hurt either.

If you use this code in your research, please cite our paper:
```
@article{kumar19bear,
  author       = {Aviral Kumar and Justin Fu and George Tucker and Sergey Levine},
  title        = {Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction},
  conference   = {NeurIPS 2019},
  url          = {http://arxiv.org/abs/1906.00949},
}
```

For any questions/issues please contact Aviral Kumar at aviralk@berkeley.edu.
