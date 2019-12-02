import random
import argparse
import habitat
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE 
import numpy as np
import progressbar
import logging

import models
from models import RL_Agent_with_env_wrapper

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
	horizon = configuration.TASK.HARIZON
	batch_size = configuration.TRAIN.BATCH_SIZE
	num_iterations = configuration.TRAIN.NUM_ITERATIONS
	step =  num_iterations//100

	logger, final_output_dir, tb_log_dir = create_logger(
        configuration, args.cfg, 'train')
	
	habitat_config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
	habitat_config.defrost()  
	habitat_config.DATASET.DATA_PATH = '/data/datasets/pointnav/gibson/v1/val/val.json.gz'
	habitat_config.DATASET.SCENES_DIR = '/data/scene_datasets/gibson'
	habitat_config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
	habitat_config.SIMULATOR.TURN_ANGLE = 45
	habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon
	habitat_config.freeze()

	environment = HabitatWrapper(configuration, habitat_config)
  	environment.reset()

  	if configuration.TRAIN.OPTIMIZER == 'adam':
		optimizer = Adam(learning_rate=configuration.TRAIN.LR)
	else:
		raise error('%s is not supported' % configuration.TRAIN.OPTIMIZER)

	if configuration.LOSS.TYPE == 'mse':
		loss_function = MSE
	else:
		raise error('%s is not supported' % configuration.LOSS.TYPE)

	agent  = models.RL_Agent(environment, optimizer, loss_function, training_embedding=False, num_actions=configuration.TASK.NUM_ACTIONS)

	num_episodes = len(environment.env.episodes)
	random_episodes_threshold = configuration.TASK.RANDOM_EPISODES_THRESHOLD
	align_model_threshold = configuration.TASK.ALIGH_MODEL_THRESHOLD

	for i in range(num_iterations):
		bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()
		for e in range(0, num_episodes):
			# Reset the enviroment
		    #logger.info('EPISODE {}'.format(e))
			agent.reset(e) #reset the environment, sets the episode-index to e

			if e < random_episodes_threshold:
				training = False
			else:
				training = True 

			if e%align_model_threshold == 1 and training:
				agent.align_target_model()
			for timestep in range(horizon):
				action = agent.sample_action()
				agent.step(action, training=training)    

		if i%(step) == 0:
			bar.update(i//step + 1)
		
		bar.finish()
		logger.info('Finished iteration [{}/{}] and start validation.'.format(i, num_iterations))
		validate(i, logger, configuration, environment, agent)

	