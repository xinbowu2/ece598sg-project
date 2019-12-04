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
	#step =  num_iterations//100

	train_scene_list = os.listdir('./data/datasets/pointnav/gibson/v1/train/content/')
	episodes_per_train_scene = configuration.TRAIN.EPISODES_PER_SCENE

	logger, final_output_dir, tb_log_dir = create_logger(
        configuration, args.cfg, 'train')
	
	habitat_config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
	habitat_config.defrost()  
	#habitat_config.DATASET.DATA_PATH = '/data/datasets/pointnav/gibson/v1/train/content'
	habitat_config.DATASET.SCENES_DIR = '/data/scene_datasets/gibson'
	habitat_config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
	habitat_config.SIMULATOR.TURN_ANGLE = 45
	habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon
	habitat_config.freeze()

	environment = HabitatWrapper(configuration, habitat_config)
	#environment.reset()
	if configuration.TRAIN.OPTIMIZER == 'adam':
		optimizer = Adam(learning_rate=configuration.TRAIN.LR)
	else:
		raise error('%s is not supported' % configuration.TRAIN.OPTIMIZER)

	if configuration.LOSS.TYPE == 'mse':
		loss_function = MSE
	else:
		raise error('%s is not supported' % configuration.LOSS.TYPE)

	agent  = RL_Agent(environment, optimizer, loss_function, training_embedding=True, num_actions=configuration.TASK.NUM_ACTIONS)

	#num_episodes = len(environment.env.episodes)
	random_episodes_threshold = configuration.TASK.RANDOM_EPISODES_THRESHOLD
	align_model_threshold = configuration.TASK.ALIGH_MODEL_THRESHOLD

	n = 0
	for i in range(num_iterations):
		bar = progressbar.ProgressBar(maxval=len(train_scene_list), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()
		random.shuffle(train_scene_list)
		for s, scene in enumerate(train_scene_list):
			habitat_config.defrost()
			habitat_config.DATASET.DATA_PATH = '/data/datasets/pointnav/gibson/v1/train/content/' + scene
			habitat_config.freeze()
			agent.environment.get_env().reconfigure(habitat_config)
		 	#agent.reset() # ??

			#num_episodes = len(agent.environment.get_env().episodes)
			#sampled_episodes = random.sample(range(0, num_episodes), episodes_per_train_scene)
			agent.environment.get_env().episodes = random.shuffle(agent.environment.get_env().episodes)

			for e in range(episodes_per_train_scene):
				agent.reset()
				if n < random_episodes_threshold:
					training = False
				else:
					print('finish filling up replay buffer and start training')
					training = True
				if n%align_model_threshold == 1 and training:
					print('align the models')
					agent.align_target_model()
				for timestep in range(horizon):
					action = agent.sample_action()
					agent.step(action, timestep=timestep, training=training)  
				
				n += 1
			
			bar.update(s+1)					


		bar.finish()
		logger.info('Finished iteration [{}/{}] and start validation.'.format(i, num_iterations))
		validate(i, logger, configuration, habitat_config, agent)
		

	
