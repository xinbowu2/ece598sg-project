import sys
sys.path.append('..')
from habitat_api_wrapper import HabitatWrapper

import habitat
from common import *
#import cv2
import numpy as np
np.random.seed(TEST_RANDOM_SEED)
import tensorflow.keras as keras
import random
random.seed(TEST_RANDOM_SEED)

def test_setup():
  #game = doom_navigation_setup(TEST_RANDOM_SEED, wad)
  #wait_idle(game, WAIT_BEFORE_START_TICS)
  config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
  config.defrost()
  config.DATASET.SPLIT = 'train_mini'
  config.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_CONTINUOUS_PLAY*10
  #config.SEED = random.randint(1, ACTION_MAX_EPOCHS)
  config.freeze()
  # print(config)

  #     0: 'move_forward',
  #     1: 'turn left',
  #     2: 'turn right'
  #     3: 'stop',

  action_mapping = {
      'w': 0,
      'a': 1,
      'd': 2,
      'f': 3
  }

  game = HabitatWrapper(config=config, action_mapping=action_mapping)

  return game

# limit memory usage
import tensorflow as tf
