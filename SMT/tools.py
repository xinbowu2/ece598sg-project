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
import matplotlib.pyplot as plt
import networkx
from networkx import *

def validate_scene_graph(training_iterations, logger, configs, final_output_dir, habitat_config, agent):
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
			action, encoder_att_weights, decoder_att_weights  = agent.sample_action(evaluating=True)

			logger.info(action)
			curr_reward, _ = agent.step(action, batch_size=None, timestep=timestep, training=False, evaluating=True)    
			#print('reward: ', curr_reward)
			episode_reward += curr_reward
			#print(timestep)
			if timestep == horizon-2:
				encoder_att_weights = tf.keras.backend.get_value(encoder_att_weights)[0]
				decoder_att_weights = tf.keras.backend.get_value(decoder_att_weights)[0]
				build_scene_graph(training_iterations, e, logger, configs, final_output_dir, encoder_att_weights, decoder_att_weights)


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


def build_scene_graph(training_iterations, episode_id, logger, configs, final_output_dir, encoder_att_weights, decoder_att_weights):
	
	assert encoder_att_weights.shape[0] == encoder_att_weights.shape[1]
	adjacency_matrix = np.zeros((encoder_att_weights.shape[0]+1, encoder_att_weights.shape[1]+1))
	adjacency_matrix[:-1,:-1] = encoder_att_weights
	adjacency_matrix[:, -1] = np.zeros((encoder_att_weights.shape[1]+1))
	adjacency_matrix[-1,:] = decoder_att_weights

	graph = networkx.from_numpy_matrix(adjacency_matrix)
	networkx.draw(graph)
	plt.draw()







