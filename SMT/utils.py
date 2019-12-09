import os
import logging
import time
import random
from pathlib import Path
from dataset import HabitatWrapper
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import progressbar
import pdb
def create_logger(cfg, cfg_name, phase='train'):
	root_output_dir = Path(cfg.OUTPUT_DIR)
	# set up logger
	if not root_output_dir.exists():
		print('=> creating {}'.format(root_output_dir))
		root_output_dir.mkdir()

	dataset = cfg.DATASET.DATASET
	model = cfg.MODEL.NAME
	cfg_name = os.path.basename(cfg_name).split('.')[0]

	final_output_dir = root_output_dir / dataset / cfg_name

	print('=> creating {}'.format(final_output_dir))
	final_output_dir.mkdir(parents=True, exist_ok=True)

	time_str = time.strftime('%Y-%m-%d-%H-%M')
	log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
	final_log_file = final_output_dir / log_file
	head = '%(asctime)-15s %(message)s'
	logging.basicConfig(filename=str(final_log_file),
						format=head)
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	console = logging.StreamHandler()
	logging.getLogger('').addHandler(console)

	tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
			(cfg_name + '_' + time_str)
	print('=> creating {}'.format(tensorboard_log_dir))
	tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

	return logger, str(final_output_dir), str(tensorboard_log_dir)

#which episodes to visualize? 
#random_policy=False means follow the policy
def visualize_trajectory(episode_indices, configs, habitat_config, agent, random_policy=False):
	horizon = configs.TEST.HORIZON
	horizon_to_resume = habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS
	habitat_config.defrost()
	habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon-1
	habitat_config.freeze()
	#environment_to_resume = agent.environment.get_env()
	agent.environment.get_env().close()
	agent.environment = HabitatWrapper(configs, habitat_config)
	sum_reward=0

	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

	for idx in episode_indices:
		# Reset the enviroment
		#print("EPISODE ", e)
		episode_reward = 0
		observations = agent.reset(idx) #reset the environment, sets the episode-index to e
		ax.imshow(tf.transpose(observations["rgb"], perm=[1,2,0]))
		#plt.show()
		#pdb.set_trace()
		plt.show(block=False)
		for timestep in range(horizon-1):
			input("Press any key to proceed: ")			
			if random_policy:
				action = random.randint(0, agent.action_size-1)				
			else:
				action, _, _ = agent.sample_action(evaluating=True)
			logger.info(action)
			reward, observations = agent.step(action, batch_size=None, timestep=timestep, training=False, evaluating=True)
			episode_reward += reward    
			print(reward)
			ax.imshow(tf.transpose(observations["rgb"], perm=[1,2,0]))
			#plt.show()
			plt.show(block=False)

		sum_reward += episode_reward 
		
		
		bar.update(10*e + 1)
		
		#agent.environment.get_env().close()
		#agent.environment.get_env()._current_episode_index = 0

	agent.environment.get_env().close()
	habitat_config.defrost()
	habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon_to_resume
	habitat_config.freeze()
	agent.environment = HabitatWrapper(configs, habitat_config)

def validate(training_iterations, logger, configs, habitat_config, agent, random_policy=False):
	batch_size = configs.TRAIN.BATCH_SIZE
	horizon = configs.TEST.HORIZON
	horizon_to_resume = habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS
	habitat_config.defrost()
	habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon-1
	habitat_config.freeze()
	#environment_to_resume = agent.environment.get_env()
	agent.environment.get_env().close()
	agent.environment = HabitatWrapper(configs, habitat_config)
	
	num_episodes = len(agent.environment.get_env().episodes)
	num_episodes = 10
	sum_reward = 0
	#step = num_episodes//100

	bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	#habitat_config.defrost()
	#habitat_config.DATASET.DATA_PATH = '/data/datasets/pointnav/gibson/v1/val_mini/val_mini.json.gz'
	#habitat_config.freeze()
	#agent.environment.get_env().reconfigure(habitat_config)
	
	#print(num_episodes)
	for e in range(0, num_episodes):
		# Reset the enviroment
		#print("EPISODE ", e)
		episode_reward = 0
		
		agent.reset(e) #reset the environment, sets the episode-index to e

		for timestep in range(horizon-1):
			if random_policy:
				action = random.randint(0, agent.action_size-1)				
			else:
				action, _, _  = agent.sample_action(evaluating=True)
			logger.info(action)
			curr_reward, _ = agent.step(action, batch_size=None, timestep=timestep, training=False, evaluating=True)    
			#print('reward: ', curr_reward)
			episode_reward += curr_reward
			#print(timestep)
		sum_reward += episode_reward 
		
		
		bar.update(10*e + 1)
		
		#agent.environment.get_env().close()
		#agent.environment.get_env()._current_episode_index = 0
	agent.environment.get_env().close()
	habitat_config.defrost()
	habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon_to_resume
	habitat_config.freeze()
	agent.environment = HabitatWrapper(configs, habitat_config)
	bar.finish()
		
	logger.info('Validation reward for %i training iterations: %f' % (training_iterations, sum_reward/num_episodes))

def validate_one_episode(training_iterations, logger, configs, habitat_config, agent, validate_episode=0, random_policy=False):
    batch_size = configs.TRAIN.BATCH_SIZE
    horizon = configs.TEST.HORIZON
    horizon_to_resume = habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS
    habitat_config.defrost()
    habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon-1
    habitat_config.freeze()
    #environment_to_resume = agent.environment.get_env()
    agent.environment.get_env().close()
    agent.environment = HabitatWrapper(configs, habitat_config)
    
    num_episodes = len(agent.environment.get_env().episodes)
    num_episodes = 1
    sum_reward = 0
    #step = num_episodes//100

    bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    #habitat_config.defrost()
    #habitat_config.DATASET.DATA_PATH = '/data/datasets/pointnav/gibson/v1/val_mini/val_mini.json.gz'
    #habitat_config.freeze()
    #agent.environment.get_env().reconfigure(habitat_config)
    
    #print(num_episodes)
    for e in range(0, num_episodes):
        # Reset the enviroment
        #print("EPISODE ", e)
        episode_reward = 0
        
        agent.reset(validate_episode) #reset the environment, sets the episode-index to e

        for timestep in range(horizon-1):
            if random_policy:
                action = random.randint(0, agent.action_size-1)             
            else:
                action, _, _  = agent.sample_action(evaluating=True)
            #print(action)
            curr_reward, _ = agent.step(action, batch_size=None, timestep=timestep, training=False, evaluating=True)    
            #print('reward: ', curr_reward)
            episode_reward += curr_reward
            #print(timestep)
        sum_reward += episode_reward 
        
        
        bar.update(10*e + 1)
        
        #agent.environment.get_env().close()
        #agent.environment.get_env()._current_episode_index = 0
    agent.environment.get_env().close()
    habitat_config.defrost()
    habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon_to_resume
    habitat_config.freeze()
    agent.environment = HabitatWrapper(configs, habitat_config)
    bar.finish()
        
    logger.info('Validation reward for %i training iterations: %f' % (training_iterations, sum_reward/num_episodes))
