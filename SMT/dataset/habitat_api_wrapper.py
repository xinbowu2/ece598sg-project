import habitat
#import error
import numpy as np 
import tensorflow as tf
#import config

class HabitatWrapper:

  def __init__(self, config, habitat_config):
    self.action_names = config.TASK.ACTION_NAMES
    self.action_mapping = {action_name : index for index, action_name in enumerate(self.action_names)}
    self.action_mapping.update({index : action_name for index, action_name in enumerate(self.action_names)})
    self.modalities = config.TASK.MODUALITIES

    self.cell_height, self.cell_width = config.TASK.CELL_HEIGHT,  config.TASK.CELL_WIDTH
    self.last_cell_x, self.last_cell_y = None, None
    self.visited_cells = []
    self.reward_rate = config.TASK.REWARD_RATE
    
    self.env = habitat.Env(config=habitat_config)

    self.observations = None
    self.prev_x, self.prev_y = None, None
    self.curr_x, self.curr_y = None, None
    self.current_action = None
    self.prev_action = None
    self.is_episode_finished = False


    
  def process_observation(self, observations):
    observations_dict = {}
    observations_dict['rgb'] = observations['rgb']/255.0
    observations_dict['rgb'] = tf.image.resize(observations_dict['rgb'],
                size=(64,64))
    #print(observations_dict['rgb'].shape)
    observations_dict['rgb'] = tf.transpose(observations_dict['rgb'], perm=[2,0,1])	
    observations_dict['prev_action'] = np.array( [0.0, 0.0, 0.0], dtype=np.float32)
    if self.prev_action:
      observations_dict['prev_action'][self.prev_action] += 1.0
    observations_dict['pose'] = np.array([self.curr_x, self.curr_y], dtype=np.float32)
    return observations_dict

  def reset(self):
    self.current_action = None
    self.prev_action = None
    self.observations = self.process_observation(self.env.reset())
    
    self.curr_x, _, self.curr_y = self.env.sim.get_agent_state().position

    self.last_cell_x, self.last_cell_y = self.curr_x, self.curr_y
    self.visited_cells = []
    self.visited_cells.append([self.curr_x,self.curr_y])
    self.is_episode_finished = self.env.episode_over
	
    return self.observations

  def get_env(self):
    return self.env

  def set_action(self, action):
    if action in self.action_mapping.keys():
      self.current_action = action
    else:
      raise error('Invalid action: %s' % action)

  def advance_action(self, tics=1, update=True):
    if update:
      if self.current_action != None:
        self.prev_action = self.current_action
        self.observations = self.process_observation(self.env.step(self.current_action))
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
        self.curr_x, _, self.curr_y = self.env.sim.get_agent_state().position
        #print('in advance action for position:', self.curr_x, self.curr_y)
        self.is_episode_finished = self.env.episode_over
	
        return self.observations

  def is_terminated(self):
    return self.is_episode_finished

  def get_prev_action(self):
    return self.prev_action

  def get_reward(self):
    '''
    if self.prev_action == 0:
      return 5.0
    '''

    collision_reward = 0.0

    if [self.curr_x, self.curr_y] == [self.prev_x, self.prev_y] and self.prev_action = 0:
      collision_reward = -1.0

    curr_cell_pos = [self.last_cell_x, self.last_cell_y]

    # calculate displacement from last cell position
    dx = self.curr_x - self.last_cell_x 
    dy = self.curr_y - self.last_cell_y
    #print('dx,dy: %f, %f'%(dx, dy))
    # calculate new cell position, if the agent enters a new cell
    if np.abs(dx) > self.cell_width/2:
      curr_cell_pos[0] += np.sign(dx)*self.cell_width

    if np.abs(dy) > self.cell_height/2:
      curr_cell_pos[1] += np.sign(dy)*self.cell_height
    
    #print(self.cell_width, ' ', self.cell_height)
    # no reward if the agent does not enter a new unvisited cell
    if curr_cell_pos == [self.last_cell_x, self.last_cell_y] or curr_cell_pos in self.visited_cells:
      self.last_cell_x = curr_cell_pos[0]
      self.last_cell_y = curr_cell_pos[1]
      return 0.0 + collision_reward

    # update new cell information
    self.visited_cells.append(curr_cell_pos)
    self.last_cell_x = curr_cell_pos[0]
    self.last_cell_y = curr_cell_pos[1]

    return self.reward_rate + collision_reward




