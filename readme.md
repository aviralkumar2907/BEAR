# BEAR (Bootstrapping Error Accumulation Reduction)

This is the code for NeurIPS 2019 paper Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction. Please refer to the project page: https://sites.google.com/view/bear-off-policyrl for details and slides explaining the algorithm.

Our code is built off of the BCQ[https://github.com/sfujim/BCQ] repository and uses many similar components. To run BEAR, please use a command like this:

```
python main.py --buffer_name=buffer_walker_300_curr_action.pkl --eval_freq=1000 --algo_name=BEAR
--env_name=Walker2d-v2 --log_dir=data_walker_BEAR/ --lagrange_thresh=10.0 
--distance_type=MMD --mode=auto --num_samples_match=5 --lamda=0.0 --version=0 
--mmd_sigma=20.0 --kernel_type=gaussian --use_ensemble_variance="False"
```

```
python main.py --buffer_name=buffer_hopper_300_curr_action.pkl --eval_freq=1000 --algo_name=BEAR
--env_name=Hopper-v2 --log_dir=data_hopper_BEAR/ --lagrange_thresh=10.0 --distance_type=MMD
--mode=auto --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=10.0 --kernel_type=laplacian --use_ensemble_variance="False"
```
**Installation Instructions**:
Please download rlkit[https://github.com/vitchyr/rlkit] and follow the instructions on the installation of the rlkit environment as supported by your machine. Please make sure to use `mujoco_py==1.50.1.56` and `mjpro150` for the MuJoCo installation. Then run the above command. Any version of PyTorch >= 1.1.0 is supported (Note: Default rlkit pytorch version is 0.4.1, but this codebase needs pytorch >= 1.1.0; Also you might need to update numpy in your system to the latest numpy version). For easy visualization, we recommmend installing viskit[https://github.com/vitchyr/viskit] and using viskit for visualization. This repository is configured to writing log-files that are compatible with viskit.  

**Algorithms Supported**:
1. BCQ (algo_name=BCQ) [Fujimoto et.al. ICML 2019]
2. TD3 (algo_name=TD3) [Fujimoto et.al. ICML 2018]
3. Behavior Cloning (algo_name=BC)
4. KL Control (algo_name=KLControl) [Jacques et.al. arxiv 2019]
5. Deep Q-learning from Demonstrations (algo_name=DQfD) [Hester et.al. 2017]

**Hyperparameter definitions**:
1. `mmd_sigma`: Standard deviation of the kernel used for MMD computation
2. `kernel_type`: (gaussian|laplacian) Kernel type used for computation of MMD
3. `num_samples_match`: Number of samples used for computing sampled MMD
4. `version`: (0|1|2): Whether to use min(0), max(1) or mean(2) of Q-values from the ensemble for policy improvement
5. `buffer_name`: Path to the buffer (prefered .pkl files, other options available in `utils.py`
6. `use_ensemble_variance`: Whether to use ensemble variance for the policy improvement step (Set to False, else can result in NaNs)
7. `lagrange_thresh`: The threshold for log of the Lagrange multiplier
8. `cloning`: Set this flag to run behaviour cloning

**Hyperparameters that generally work well (for BEAR, across environments)**:
1. `mmd_sigma=10.0`, `kernel_type=laplacian`, `num_samples_match=5`, `version=0 or 2`, `lagrange_thresh=10.0`, `mode=auto`
2. `mmd_sigma=20.0`, `kernel_type=gaussian`, `num_samples_match=5`, `version=0 or 2`, `lagrange_thresh=10.0`, `mode=auto`

We have removed ensembles from this version, and we just use a minimum/average over 2 Q-functions, without an ensemble-based conservative estimate based on sample variance. This is because we didn't find ensemble variance to in general provide benefits, although it doesn't hurt either. However, the code for ensembles is present in `EnsembleCritic` in the file `algos.py`. Also, please set `use_ensemble_variance=True` to use ensembles in the BEAR algorithm.

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
