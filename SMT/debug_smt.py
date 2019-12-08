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
from utils import create_logger, validate, visualize_trajectory

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

	train_scene_list = os.listdir('./data/datasets/pointnav/gibson/v1/train_mini/content/')
	#train_scene_list =['']
	#episodes_per_train_scene = configuration.TRAIN.EPISODES_PER_SCENE

	logger, final_output_dir, tb_log_dir = create_logger(
        configuration, args.cfg, 'train')

	logger.info('TESTING FOR GPU: ', tf.test.is_gpu_available())

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

	logger.info(configuration)
	logger.info(habitat_config)
	#iterator = environment.episode_iterator.deepcopy()
	if configuration.TRAIN.OPTIMIZER == 'adam':
		optimizer = Adam(learning_rate=configuration.TRAIN.LR)
	else:
		raise error('%s is not supported' % configuration.TRAIN.OPTIMIZER)

	if configuration.LOSS.TYPE == 'mse':
		loss_function = MSE
	else:
		raise error('%s is not supported' % configuration.LOSS.TYPE)

	agent  = RL_Agent(environment, optimizer, loss_function, batch_size=batch_size, training_embedding=False, num_actions=configuration.TASK.NUM_ACTIONS)
	
	if configuration.TRAIN.RESUME:
		agent.trace_model()
		agent.load_weights(final_output_dir + '/checkpoints/' + configuration.MODEL.CHECKPOINT)
		logger.info('loaded checkpoint: %s'%configuration.MODEL.CHECKPOINT)
	#num_episodes = len(environment.env.episodes)
	
	i = 0	
	
	#logger.info('Rewards per Episode Achieved by Random Policy: ')
	#validate(i, logger, configuration, habitat_config, agent, random_policy=True)
	
	logger.info('Rewards per Episode Achieved by Learned Policy: ')
	#visualize_trajectory([12], configuration, habitat_config, agent, random_policy=True) 
	validate(i, logger, configuration, habitat_config, agent, random_policy=False)	
