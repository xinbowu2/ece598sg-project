import habitat
import numpy as np

class HabitatWrapper:

  def __init__(self, config):
    self.env = habitat.Env(config=config)

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
    curr_x, curr_y = self.game_variables['position']

    self.last_cell_x, self.last_cell_y = curr_x, curr_y
    self.visited_cells.append([curr_x,curr_y])
    self.is_episode_finished = self.env.episode_over()

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
        self.is_episode_finished = self.env.episode_over()

  def is_episode_finished(self):
    return self.is_episode_finished

  def get_prev_action(self):
    return self.prev_action

  def get_reward(self):
    curr_x, curr_y = self.game_variables['position']
    curr_cell_pos = [self.last_cell_x, self.last_cell_y]

    # calculate displacement from last cell position
    dx = curr_x - self.last_cell_x 
    dy = curr_y - self.last_cell_y

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

  

