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


# num_of_episodes = config.TRAINING.NUM_EPISODES
# modalities = config.DATASETS.MODALITIES
# batch_size = config.TRAINING.BATCH_SIZE
# timesteps_per_episode = config.TRAINING.TIMESTEPS_PER_EPISODE

horizon = 100

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
agent  = models.RL_Agent(enviroment, optimizer, loss_function, training_embedding=True)

num_episodes = len(enviroment.episodes)
random_episodes_threshold = 500
align_model_threshold = 5

for e in range(0, num_of_episodes):
	# Reset the enviroment
	agent.reset(e) #reset the environment, sets the episode-index to e

	if e < random_episodes_threshold:
		training = False
	else:
		training = True 

	bar = progressbar.ProgressBar(maxval=horizon/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()

	if e%align_model_threshold == 0:
		agent.align_target_model()
		
	for time_step in range(horizon):
		action = agent.sample_action()
		agent.step(action)    

		if timestep%10 == 0:
			bar.update(timestep/10 + 1)
	
	bar.finish()
