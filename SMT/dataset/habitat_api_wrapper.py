import habitat
import numpy as np

class HabitatWrapper:

  def __init__(self, config, habitat_config):
    self.env = habitat.Env(config=habitat_config)

    self.action_mapping = config['DATASET'].ACTION_MAPPING
    self.action_mapping.update({action_name : index for index, action_name in action_mapping.items()})
    self.modalities = config['TASK'].MODUALITIES

    self.cell_height, self.cell_width = config['TASK'].CELL_HEIGHT,  config['TASK'].CELL_WIDTH
    self.last_cell_x, self.last_cell_y = None, None
    self.visited_cells = []
    self.reward_rate = config['TASK'].REWARD_RATE

    self.observations = None
    self.current_action = None
    self.prev_action = None
    self.is_episode_finished = False

  def reset(self):
    self.observations = self.env.reset()
    self.curr_x, _, self.curr_y = self.env.sim.get_agent_state().position

    self.last_cell_x, self.last_cell_y = self.curr_x, self.curr_y
    self.visited_cells.append([curr_x,curr_y])
    self.is_episode_finished = self.env.episode_over()

    return self.observations

  def get_state(self):
    state = self.observations.copy().add('prev_action', self.prev_action)

    return self.observations

  def set_action(self, action):
    if action in action_mapping.keys():
      self.current_action = action
    else:
      raise error('Invalid action: %s' % action)

  def advance_action(self, tics=1, update=True):
    if update:
      if self.current_action != None:
        self.observations = self.env.step(self.action_mapping[self.current_action])
        self.prev_action = self.current_action
        self.curr_x, _, self.curr_y = self.env.sim.get_agent_state().position
        self.is_episode_finished = self.env.episode_over()

        return self.observations

  def is_episode_finished(self):
    return self.is_episode_finished

  def get_prev_action(self):
    return self.prev_action

  def get_reward(self):
    curr_cell_pos = [self.last_cell_x, self.last_cell_y]

    # calculate displacement from last cell position
    dx = self.curr_x - self.last_cell_x 
    dy = self.curr_y - self.last_cell_y

    # calculate new cell position, if the agent enters a new cell
    if np.abs(dx) > self.cell_width:
      curr_cell_pos[0] += np.sign(dx)*self.cell_width

    if np.abs(dy) > self.cell_height:
      curr_cell_pos[1] += np.sign(dy)*self.cell_height

    # no reward if the agent does not enter a new unvisited cell
    if current_cell_pos == [self.last_cell_x, self.last_cell_y] or current_cell_pos in self.visited_cells:
      return 0

    # update new cell information
    visited_cells.append(current_cell_pos)
    self.last_cell_x = current_cell_pos[0]
    self.last_cell_y = current_cell_pos[1]

    return self.reward_rate

'''
if __name__ == '__main__':
  import random

  from config import config
  from config import update_config

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
    update_config(config, args)

    return args

  args = parse_args()

  num_steps = 100
  actions = [random.choice([0,1,2,3 for i in range(num_steps)])]


  habitat_config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
  habitat_config.defrost()  
  habitat_config.DATASET.DATA_PATH = '../data/datasets/pointnav/gibson/v1/val/val.json.gz'
  habitat_config.DATASET.SCENES_DIR = '../data/scene_datasets/gibson'
  habitat_config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR'] 
  habitat_config.SIMULATOR.TURN_ANGLE = 30
  habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_CONTINUOUS_PLAY*64

  habitat_config.freeze()

  env = HabitatWrapper(config, habitat_config)
  env.reset()
  
  for i in range(num_steps):
    env.set_action(actions[i])
    env.advance_action()
'''

    

