import habitat
import random

num_epochs = 20
batch_size = 64
episodes_per_house = 10
print("Begin Training")

action_mapping = {      
  0: 'move_forward',
  1: 'turn left',
  2: 'turn right',
  3: 'stop'
}


config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
config.defrost()  
config.DATASET.DATA_PATH = '../data/datasets/pointnav/gibson/v1/val/val.json.gz'
config.DATASET.SCENES_DIR = '../data/scene_datasets/gibson'
config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
config.SIMULATOR.TURN_ANGLE = 30
#config.TASK.SENSORS = ["PROXIMITY_SENSOR"]
config.ENVIRONMENT.MAX_EPISODE_STEPS = 500
config.freeze()
environment = habitat.Env(config=config)



for e in range(0, len(environment.episodes)):
	obs = environment.reset()['rgb']/255.0
	reward = 0
	terminated = False

	for iter_ in range(episodes_per_house):
		while not environment.episode_over:
