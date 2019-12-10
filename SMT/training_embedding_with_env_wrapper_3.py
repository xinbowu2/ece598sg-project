import random
import argparse
import habitat
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE 
import numpy as np
import progressbar
import logging
import os

#import models
from models.rl_agent_with_env_wrapper import RL_Agent

import config
from config import update_config
from config import configuration
from dataset import HabitatWrapper  
from utils import create_logger, validate

def parse_args():
  parser = argparse.ArgumentParser(description='Train segmentation network')
  
  parser.add_argument('--cfg',
                      help='experiment configure file name',
                      required=True,
                      type=str)
  parser.add_argument('opts',
                      help="Modify config options using the command-line",
                      default=None,
                      nargs=argparse.REMAINDER)

  args = parser.parse_args()
  update_config(configuration, args)

  return args

if __name__ == '__main__':
	args = parse_args()
	horizon = configuration.TASK.HORIZON
	batch_size = configuration.TRAIN.BATCH_SIZE
	num_iterations = configuration.TRAIN.NUM_ITERATIONS
	validate_save_freq = configuration.TRAIN.VALIDATE_SAVE_FREQ
	#step =  num_iterations//100
	#episilon = 0.4	

	train_scene_list = os.listdir('./data/datasets/pointnav/gibson/v1/train_mini/content/')
	#train_scene_list =['']
	#episodes_per_train_scene = configuration.TRAIN.EPISODES_PER_SCENE

	logger, final_output_dir, tb_log_dir = create_logger(
        configuration, args.cfg, 'train')
	
	habitat_config = habitat.get_config(config_file='datasets/pointnav/gibson.yaml')
	habitat_config.defrost()  
	habitat_config.DATASET.SPLIT = 'train_mini'
	#habitat_config.DATASET.DATA_PATH = '/data/datasets/pointnav/gibson/v1/train/content/Aridan.json.gz'
	#habitat_config.DATASET.SCENES_DIR = '/data/scene_datasets/gibson_1'
	habitat_config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
	habitat_config.SIMULATOR.TURN_ANGLE = 30
	habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon-1
	habitat_config.freeze()
	#print(habitat_config)
	environment = HabitatWrapper(configuration, habitat_config)
	environment.reset()
	#iterator = environment.episode_iterator.deepcopy()
	if configuration.TRAIN.OPTIMIZER == 'adam':
		optimizer = Adam(learning_rate=configuration.TRAIN.LR)
	else:
		raise error('%s is not supported' % configuration.TRAIN.OPTIMIZER)

	if configuration.LOSS.TYPE == 'mse':
		loss_function = MSE
	else:
		raise error('%s is not supported' % configuration.LOSS.TYPE)

	agent  = RL_Agent(environment, optimizer, loss_function, batch_size=batch_size, training_embedding=True, num_actions=configuration.TASK.NUM_ACTIONS)
	
	if configuration.TRAIN.RESUME:
		agent.trace_model()
		agent.load_weights(final_output_dir + '/checkpoints/' + configuration.MODEL.CHECKPOINT)

	#num_episodes = len(environment.env.episodes)
	random_episodes_threshold = configuration.TASK.RANDOM_EPISODES_THRESHOLD
	align_model_threshold = configuration.TASK.ALIGH_MODEL_THRESHOLD
	
	logger.info(configuration)
	logger.info(habitat_config)

	n = 0

	step = len(agent.environment.get_env().episodes)//100
	print('step', step)
	episodes_per_train_scene  = len(agent.environment.get_env().episodes)
	print(len(agent.environment.get_env().episodes))
	for i in range(num_iterations):
		bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()	
		 	#agent.reset() # ??

		num_episodes = len(agent.environment.get_env().episodes)
		sampled_episodes = random.sample(range(0, num_episodes), num_episodes)
		for e, episode_id in enumerate(sampled_episodes):
			#print(agent.environment.get_env()._current_episode_index)
			agent.reset(episode_id)
			if n < random_episodes_threshold:
				training = False
			elif not training:
				logger.info('finish filling up replay buffer and start training')
				training = True
			#if n%align_model_threshold == 1 and training:
			for timestep in range(horizon-1):
				action = agent.sample_action(training=training)
				#print(action)
				#if training:
					#print('action after training: ', action)
				agent.step(action, timestep=timestep, batch_size=batch_size, training=training)  
		
			n += 1
			if n%10 == 0:
				print(n)
			if training and n%align_model_threshold == 0:
				print('align models')
				agent.align_target_model()
			#agent.environment.get_env().episode_iterator = iterator
			#agent.environment.get_env().close()
			#agent.environment.get_env().reconfigure(habitat_config)
			#agent.environment = HabitatWrapper(configuration, habitat_config)
			bar.update(e/step+1)	
				
			if (n+1)%validate_save_freq == 0 and training:
				logger.info('saving checkpoint after episodes %i'%n)
				agent.save_weights(final_output_dir + '/checkpoints/cp-episode{}.ckpt'.format(n))
				validate(i, logger, configuration, habitat_config, agent)

		bar.finish()
		logger.info('Finished iteration [{}/{}] and start validation.'.format(i, num_iterations))
		logger.info('saving checkpoint %i'%i)
		agent.save_weights(final_output_dir + '/checkpoints/cp-{}.ckpt'.format(i))
		validate(i, logger, configuration, habitat_config, agent)
		

	
