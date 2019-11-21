import habitat
import numpy as np

class HabitatWrapper:

  def __init__(self, config, action_mapping):
    self.env = habitat.Env(config=config)
    self.action_mapping = action_mapping
    self.action_mapping.update({action_name : index for index, action_name in action_mapping.items()})

    self.cell_height, self.cell_width = config['TASK'].CELL_HEIGHT,  config['TASK'].CELL_WIDTH
    self.last_cell_x, self.last_cell_y = 0, 0
    self.visited_cells = []

    self.reward_rate = config['TASK'].REWARD_RATE

    self.current_action = None
    self.last_action = None
    self.game_state = GameState(self.env)
    self.is_episode_finished = False

  def reset(self):
    observations = self.env.reset()
    self.game_state.update(observations)
    self.last_cell_x, self.last_cell_y = 0, 0
    self.visited_cells.append([0,0])
    self.is_episode_finished = self.env.episode_over()

  def get_state(self):
    return self.game_state

  def set_action(self, action):
    if action in action_mapping.keys():
      self.current_action = action
    else:
      raise error('Invalid action: %s' % action)

  def advance_action(self, tics=1, update=True):
    if update:
      if self.current_action != None:
        observations = self.env.step(self.action_mapping[self.current_action])
        self.game_state.update(observations)
        self.last_action = self.current_action
        self.is_episode_finished = self.env.episode_over()

  def is_episode_finished(self):
    return self.is_episode_finished

  def get_last_action(self):
    return self.last_action

  def replay_episode(self, lmp):
    raise NotImplementedError

  def new_episode(self):
    raise NotImplementedError

  def get_reward(self):
    curr_x, curr_y = self.game_variables['position']
    curr_cell_pos = [self.last_cell_x, self.last_cell_y]

    dx = curr_x - self.last_cell_x
    dy = curr_y - self.last_cell_y

    if np.abs(dx) > self.cell_width:
      curr_cell_pos[0] += np.sign(dx)*self.cell_width

    if np.abs(dy) > self.cell_height:
      curr_cell_pos[1] += np.sign(dy)*self.cell_height

    if current_cell_pos == [self.last_cell_x, self.last_cell_y] or current_cell_pos in self.visited_cells:
      return 0

    visited_cells.append(current_cell_pos)
    self.last_cell_x = current_cell_pos[0]
    self.last_cell_y = current_cell_pos[1]

    return self.reward_rate



class GameState:
  def __init__(self, env):
    self.env = env

    self.number = None
    self.tic = None
    self.game_variables = None
    self.screen_buffer = None
    self.depth_buffer = None
    self.labels_buffer = None
    self.automap_buffer = None
    self.labels = None

  def update(self, observations):
    self.screen_buffer = observations['rgb']
    # TO DO: self.game_state.game_variables = ??
  

