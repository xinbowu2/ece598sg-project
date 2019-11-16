import tensorflow as tf
import habitat
import models

# set up models 

modalities = ['image']
modality_dim = {'image': 64}
reduce_factor = 4
observation_dim = 128

scene_memory = models.SceneMemory(modalities, modality_dim, reduce_factor, 
  observation_dim)



# 256x256x3



# set up environment
config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
config.defrost()  
config.DATASET.DATA_PATH = '../data/datasets/pointnav/gibson/v1/val/val.json.gz'
config.DATASET.SCENES_DIR = '../data/scene_datasets/gibson'
config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
config.SIMULATOR.TURN_ANGLE = 30
config.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_CONTINUOUS_PLAY*64

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

current_x = env.reset()['rgb']/255.0



