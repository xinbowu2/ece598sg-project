import random
import argparse
import habitat

import config
from config import update_config
from config import configuration
from dataset import HabitatWrapper  

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

  num_steps = 200
  actions = [random.choice([0,1,2]) for i in range(num_steps)]


  habitat_config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
  habitat_config.defrost()  
  habitat_config.DATASET.DATA_PATH = '../data/datasets/pointnav/gibson/v1/val/val.json.gz'
  habitat_config.DATASET.SCENES_DIR = '../data/scene_datasets/gibson'
  habitat_config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
  habitat_config.SIMULATOR.TURN_ANGLE = 45
  habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = 60
  habitat_config.freeze()

  env = HabitatWrapper(configuration, habitat_config)
  env.reset()
  
  for i in range(num_steps):
    if env.is_terminated():
      env.reset()
      print('reset environment')
    else:
      print('action: ', env.action_mapping[actions[i]])
      print('before step position: (%f, %f)'%(env.curr_x, env.curr_y))
      env.set_action(actions[i])
      env.advance_action()
      print('after step position: (%f, %f), before reward last cell position: (%f, %f), current_reward: %i' % (env.curr_x, env.curr_y, env.last_cell_x, env.last_cell_y, env.get_reward()))
      print('after reward last cell position: (%f, %f)'%(env.last_cell_x, env.last_cell_y))
