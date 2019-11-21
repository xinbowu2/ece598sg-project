import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
#from IPython.display import clear_output
from collections import deque
import progressbar

from models import RL_Agent
from datasets.habitat_api_wrapper import HabitatWrapper


num_of_episodes = config.TRAINING.NUM_EPISODES
modalities = config.DATASETS.MODALITIES
batch_size = config.TRAINING.BATCH_SIZE
timesteps_per_episode = config.TRAINING.TIMESTEPS_PER_EPISODE

environment = HabitatWrapper(config)
agent  = models.RL_Agent(config)

optimizer = Adam(learning_rate=0.001)

for e in range(0, num_of_episodes):
    # Reset the enviroment
    state = enviroment.reset()
    '''
    if 'image' in modalities:
    	state['image'] = np.reshape(state, [1, 1])
   	'''
    
    # Initialize variables
    reward = 0
    terminated = False
    
    bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for timestep in range(timesteps_per_episode):
        # Run Action
        action = agent.act(state)
        
        # Take action    
        next_state, reward, terminated, info = enviroment.step(action) 
        #next_state = np.reshape(next_state, [1, 1])
        agent.store(state, action, reward, next_state, terminated)
        
        state = next_state
            
        if len(agent.expirience_replay) > batch_size:
            agent.retrain(batch_size)
        
        if timestep%10 == 0:
            bar.update(timestep/10 + 1)
    
    bar.finish()
