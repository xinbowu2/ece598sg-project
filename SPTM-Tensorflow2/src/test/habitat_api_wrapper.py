import habitat
from common import *

class HabitatWrapper:

  def __init__(self, config, action_mapping):
    self.env = habitat.Env(config=config)
    self.action_mapping = action_mapping
    self.action_mapping.update({action_name : index for index, action_name in action_mapping.items()})

    self.current_action = None
    self.last_action = None
    self.game_state = GameState(self.env)
    self.episode_finished = False

  def reset(self):
    observation = self.env.reset()
    position = self.env.sim.get_agent_state().position
    self.game_state.update(observation, position)
    self.episode_finished = self.env.episode_over()

  def get_state(self):
    return self.game_state

  def set_action(self, action):
    if action in self.action_mapping.keys():
      self.current_action = action
    else:
      raise('Invalid action: %s' % action)

  def advance_action(self, tics=1, update=True):
    if update:
      if self.current_action != None:
        observation = center_crop_resize(env.step(self.action_mapping[self.current_action])['rgb']/255.0, 256)
        position = self.env.sim.get_agent_state().position
        self.game_state.update(observation, position)
        self.last_action = self.current_action
        self.episode_finished = self.env.episode_over()

  def is_episode_finished(self):
    return self.is_episode_finished

  def get_last_action(self):
    return self.last_action

  def replay_episode(self, lmp):
    raise NotImplementedError

  def new_episode(self):
    raise NotImplementedError

class GameState:
  def __init__(self, env):
    self.env = env

    self.number = None
    self.tic = None
    self.game_variables = None
    self.position = None
    self.screen_buffer = None
    self.depth_buffer = None
    self.labels_buffer = None
    self.automap_buffer = None
    self.labels = None
    self.goal_position = None
    self.goal_observation = None

  def update(self, observations, position):
    self.screen_buffer = observations['rgb']/255.0
    self.position = [position[2], position[0]]  

