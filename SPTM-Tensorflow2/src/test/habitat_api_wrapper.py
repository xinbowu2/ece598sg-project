import habitat

class HabitatWrapper:

  def __init__(self, config, action_mapping):
    self.env = habitat.Env(config=config)
    self.action_mapping = action_mapping
    self.current_action_index = None
    self.last_action_index = None
    self.game_state = GameState(self.env)
    self.is_episode_finished = False

  def reset(self):
    observations = self.env.reset()
    self.game_state.update(observations)
    self.is_episode_finished = self.env.episode_over()

  def get_state(self):
    return self.game_state

  def set_action(self, action_index):
    if action_index in action_mapping.keys():
      self.current_action_index = action_index
    else:
      raise error('Invalid action index: %s' % action_index)

  def advance_action(self, tic=1, update=True):
    if update:
      if self.current_action_index != None:
        observations = self.env.step(self.current_action_index)
        self.game_state.update(observations)
        self.last_action_index = self.current_action_index
        self.is_episode_finished = self.env.episode_over()

  def is_episode_finished(self):
    return self.is_episode_finished

  def get_last_action(self):
    return self.last_action_index

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
    self.screen_buffer = None
    self.depth_buffer = None
    self.labels_buffer = None
    self.automap_buffer = None
    self.labels = None

  def update(self, observations):
    self.screen_buffer = observations['rgb']
    # TO DO: self.game_state.game_variables = ??
  

