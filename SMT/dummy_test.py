import tensorflow as tf
import habitat
import models
import random
import pdb
import numpy as np
# set up models 
tf.keras.backend.set_floatx('float32')
num_actions = 3

# set up environment
config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
config.defrost()  
config.DATASET.DATA_PATH = '../data/datasets/pointnav/gibson/v1/val/val.json.gz'
config.DATASET.SCENES_DIR = '../data/scene_datasets/gibson'
config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
config.SIMULATOR.TURN_ANGLE = 30
config.ENVIRONMENT.MAX_EPISODE_STEPS = 100
#config.SIMULATOR.SCENE = "data/scene_datasets/gibson/Lynchburg.glb"

config.freeze()
env = habitat.Env(config=config)
r = random.randint(1, len(env.episodes))
env._current_episode = env.episodes[r]
print("ENVIRONMENT EPISODES", len(env.episodes))
action_mapping = {      
      0: 'move_forward',
      1: 'turn left',
      2: 'turn right',
      3: 'stop'
}

i=0
current_x = env.reset()['rgb']/255.0 
while True:    
    action_index = random.randint(0, 2)
    current_x = env.step(action_index)['rgb']/255.0
    if env.episode_over:
        break
    print(i)
    i+=1
