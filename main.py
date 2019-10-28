import gym
import numpy as np
import torch
import argparse
import os

import utils
import DDPG
import algos
import TD3
from logger import logger, setup_logger
from logger import create_stats_ordered_dict
import point_mass

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
	avg_reward = 0.
	all_rewards = []
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		cntr = 0
		while ((not done)):
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += reward
			cntr += 1
		all_rewards.append(avg_reward)
	avg_reward /= eval_episodes
	for j in range(eval_episodes-1, 1, -1):
		all_rewards[j] = all_rewards[j] - all_rewards[j-1]

	all_rewards = np.array(all_rewards)
	std_rewards = np.std(all_rewards)
	median_reward = np.median(all_rewards)
	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward, std_rewards, median_reward

def evaluate_policy_discounted(policy, eval_episodes=10):
	avg_reward = 0.
	all_rewards = []
	gamma = 0.99
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		cntr = 0
		gamma_t = 1
		while ((not done)):
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += (gamma_t * reward)
			gamma_t = gamma * gamma_t
			cntr += 1
		all_rewards.append(avg_reward)
	avg_reward /= eval_episodes
	for j in range(eval_episodes-1, 1, -1):
		all_rewards[j] = all_rewards[j] - all_rewards[j-1]

	all_rewards = np.array(all_rewards)
	std_rewards = np.std(all_rewards)
	median_reward = np.median(all_rewards)
	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward, std_rewards, median_reward




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="Hopper-v2")                          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                                      # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="Robust")                          # Prepends name to filename.
	parser.add_argument("--eval_freq", default=5e3, type=float)                     # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)         # Max time steps to run environment for
	parser.add_argument("--buffer_name", default=None, type=str)            # the path to the buffer file
	parser.add_argument("--version", default='0', type=str)                 # Basically whether to do min(Q), max(Q), mean(Q) over multiple Q networks for policy updates
	parser.add_argument("--lamda", default=0.5, type=float)                 # Unused parameter -- please ignore 
	parser.add_argument("--threshold", default=0.05, type=float)            # Unused parameter -- please ignore
	parser.add_argument('--use_bootstrap', default=False, type=bool)        # Whether to use bootstrapped ensembles or plain ensembles
	parser.add_argument('--algo_name', default="OursBCQ", type=str)         # Which algo to run (see the options below in the main function)
	parser.add_argument('--mode', default='hardcoded', type=str)            # Whether to do automatic lagrange dual descent or manually tune coefficient of the MMD loss (prefered "auto")
	parser.add_argument('--num_samples_match', default=10, type=int)        # number of samples to do matching in MMD
	parser.add_argument('--mmd_sigma', default=10.0, type=float)            # The bandwidth of the MMD kernel parameter
	parser.add_argument('--kernel_type', default='laplacian', type=str)     # kernel type for MMD ("laplacian" or "gaussian")
	parser.add_argument('--lagrange_thresh', default=10.0, type=float)      # What is the threshold for the lagrange multiplier
	parser.add_argument('--distance_type', default="MMD", type=str)         # Distance type ("KL" or "MMD")
	parser.add_argument('--log_dir', default='./data_hopper/', type=str)    # Logging directory
	parser.add_argument('--use_ensemble_variance', default='True', type=str)       # Whether to use ensemble variance or not
	parser.add_argument('--use_behaviour_policy', default='False', type=str)       
	parser.add_argument('--cloning', default="False", type=str)
	parser.add_argument('--num_random', default=10, type=int)
	parser.add_argument('--margin_threshold', default=10, type=float)		# for DQfD baseline
	args = parser.parse_args()

	# Use any random seed, and not the user provided seed
	seed = np.random.randint(10, 1000)
	algo_name = args.algo_name
	
	file_name = algo_name + "_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_0.1" % (args.env_name, str(seed), str(args.version), str(args.lamda), str(args.threshold), str(args.use_bootstrap), str(args.mode),\
		 str(args.kernel_type), str(args.num_samples_match), str(args.mmd_sigma), str(args.lagrange_thresh), str(args.distance_type), str(args.use_behaviour_policy), str(args.num_random))
	buffer_name = args.buffer_name
	print ("---------------------------------------")
	print ("Settings: " + file_name)
	print ("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.env_name == 'Multigoal-v0':
		env = point_mass.MultiGoalEnv(distance_cost_coeff=10.0)
	else:
		env = gym.make(args.env_name)

	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	print (state_dim, action_dim)
	print ('Max action: ', max_action)

	variant = dict(
		algorithm=algo_name,
		version=args.version,
		env_name=args.env_name,
		seed=seed,
		lamda=args.lamda,
		threshold=args.threshold,
		use_bootstrap=str(args.use_bootstrap),
		bootstrap_dim=4,
		delta_conf=0.1,
		mode=args.mode,
		kernel_type=args.kernel_type,
		num_samples_match=args.num_samples_match,
		mmd_sigma=args.mmd_sigma,
		lagrange_thresh=args.lagrange_thresh,
		distance_type=args.distance_type,
		use_ensemble_variance=args.use_ensemble_variance,
		use_data_policy=args.use_behaviour_policy,
		num_random=args.num_random,
		margin_threshold=args.margin_threshold,
	)
	setup_logger(file_name, variant=variant, log_dir=args.log_dir + file_name)

	if algo_name == 'BCQ':
		policy = algos.BCQ(state_dim, action_dim, max_action)
	elif algo_name == 'TD3':
		policy = TD3.TD3(state_dim, action_dim, max_action)
	elif algo_name == 'BC':
		policy = algos.BCQ(state_dim, action_dim, max_action, cloning=True)
	elif algo_name == 'DQfD':
		policy = algos.DQfD(state_dim, action_dim, max_action, lambda_=args.lamda, margin_threshold=float(args.margin_threshold))
	elif algo_name == 'KLControl':
		policy = algos.KLControl(2, state_dim, action_dim, max_action)
	elif algo_name == 'BEAR':
		policy = algos.BEAR(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
			version=args.version,
			lambda_=float(args.lamda),
			threshold=float(args.threshold),
			mode=args.mode,
			num_samples_match=args.num_samples_match,
			mmd_sigma=args.mmd_sigma,
			lagrange_thresh=args.lagrange_thresh,
			use_kl=(True if args.distance_type == "KL" else False),
			use_ensemble=(False if args.use_ensemble_variance == "False" else True),
			kernel_type=args.kernel_type)
	elif algo_name == 'BEAR_IS':
		policy = algos.BEAR_IS(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
			version=args.version,
			lambda_=float(args.lamda),
			threshold=float(args.threshold),
			mode=args.mode,
			num_samples_match=args.num_samples_match,
			mmd_sigma=args.mmd_sigma,
			lagrange_thresh=args.lagrange_thresh,
			use_kl=(True if args.distance_type == "KL" else False),
			use_ensemble=(False if args.use_ensemble_variance == "False" else True),
			kernel_type=args.kernel_type)

	# Load buffer
	replay_buffer = utils.ReplayBuffer()
	if args.env_name == 'Multigoal-v0':
		replay_buffer.load_point_mass(buffer_name, bootstrap_dim=4, dist_cost_coeff=0.01)
	else:
		replay_buffer.load(buffer_name, bootstrap_dim=4)
	
	evaluations = []

	episode_num = 0
	done = True 

	training_iters = 0
	while training_iters < args.max_timesteps: 
		pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))

		ret_eval, var_ret, median_ret = evaluate_policy(policy)
		evaluations.append(ret_eval)
		np.save("./results/" + file_name, evaluations)

		training_iters += args.eval_freq
		print ("Training iterations: " + str(training_iters))
		logger.record_tabular('Training Epochs', int(training_iters // int(args.eval_freq)))
		logger.record_tabular('AverageReturn', ret_eval)
		logger.record_tabular('VarianceReturn', var_ret)
		logger.record_tabular('MedianReturn', median_ret)
		logger.dump_tabular()
