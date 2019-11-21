import tensorflow as tf
import habitat
import models
import random
import pdb
import numpy as np
# set up models 
tf.keras.backend.set_floatx('float32')
modalities = ['image']
modality_dim = {'image': 64}
reduce_factor = 4
observation_dim = 128
depth = 128
num_heads = 8
d_model = num_heads*depth

num_actions = 4

scene_memory = models.SceneMemory(modalities=modalities, modality_dim=modality_dim, observation_dim=observation_dim)

policy_network = models.AttentionPolicyNet(num_actions, d_model, num_heads=num_heads)


# 256x256x3



# set up environment
config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
config.defrost()  
config.DATASET.DATA_PATH = '../data/datasets/pointnav/gibson/v1/val/val.json.gz'
config.DATASET.SCENES_DIR = '../data/scene_datasets/gibson'
config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
config.SIMULATOR.TURN_ANGLE = 30
config.ENVIRONMENT.MAX_EPISODE_STEPS = 100

config.freeze()
env = habitat.Env(config=config)
r = random.randint(1, len(env.episodes))
env._current_episode = env.episodes[r]

action_mapping = {      
      0: 'move_forward',
      1: 'turn left',
      2: 'turn right',
      3: 'stop'
}

from tensorflow.keras.optimizers import Adam
optimizer = Adam()
from tensorflow.keras.losses import MSE 
loss_function = MSE
from models import RL_Agent

agent = RL_Agent(env, optimizer, loss_function)
agent.reset(109)

agent.step(0)

observations = {}
current_x = env.reset()['rgb']/255.0
observations['image'] = current_x

current_embedding, mem = scene_memory(observations)
new_x = env.step(1)['rgb']/255.0
observations['image'] = new_x
current_embedding, mem = scene_memory(observations)

