import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
#from IPython.display import clear_output
from collections import deque
import progressbar

from models import RL_Agent
from tensorflow.keras.losses import MSE 
#from datasets.habitat_api_wrapper import HabitatWrapper
import habitat
import models
import pdb
# num_of_episodes = config.TRAINING.NUM_EPISODES
# modalities = config.DATASETS.MODALITIES
# batch_size = config.TRAINING.BATCH_SIZE
# timesteps_per_episode = config.TRAINING.TIMESTEPS_PER_EPISODE

horizon = 10
config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
config.defrost()  
config.DATASET.DATA_PATH = '../data/datasets/pointnav/gibson/v1/val/val.json.gz'
config.DATASET.SCENES_DIR = '../data/scene_datasets/gibson'
config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
config.SIMULATOR.TURN_ANGLE = 30
config.ENVIRONMENT.MAX_EPISODE_STEPS = horizon

config.freeze()
environment = habitat.Env(config=config)

#environment = HabitatWrapper(config)

optimizer = Adam(learning_rate=5e-4)
loss_function = MSE
agent  = models.RL_Agent(environment, optimizer, loss_function, training_embedding=False, num_actions=3)

num_episodes = len(environment.episodes)
random_episodes_threshold = 10
align_model_threshold = 20

for e in range(0, num_episodes):
	# Reset the enviroment
	print("EPISODE ", e)
	agent.reset(e) #reset the environment, sets the episode-index to e

	if e < random_episodes_threshold:
		training = False
	else:
		training = True 

	bar = progressbar.ProgressBar(maxval=horizon/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()

	if e%align_model_threshold == 1 and training:
		agent.align_target_model()
	for timestep in range(horizon):
		action = agent.sample_action()
		agent.step(action, training=training)    

		if timestep%10 == 0:
			bar.update(timestep/10 + 1)
	
	bar.finish()

