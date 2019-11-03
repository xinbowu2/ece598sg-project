import sys
sys.path.append('..')
from habitat_api_wrapper import HabitatWrapper

from common import *
import cv2
import numpy as np
np.random.seed(TEST_RANDOM_SEED)
import keras
import random
random.seed(TEST_RANDOM_SEED)

def test_setup(wad):
  #game = doom_navigation_setup(TEST_RANDOM_SEED, wad)
  #wait_idle(game, WAIT_BEFORE_START_TICS)
  config = habitat.get_config(config_file='tasks/pointnav_gibson.yaml')
  config.defrost()  
  config.DATASET.DATA_PATH = '../data/datasets/pointnav/gibson/v1/val/val.json.gz'
  config.DATASET.SCENES_DIR = '../data/scene_datasets/gibson'
  config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
  config.SIMULATOR.TURN_ANGLE = 30
  config.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_CONTINUOUS_PLAY*64
  config.freeze()
  action_mapping = {
      0: 'stop',
      1: 'move_forward',
      2: 'turn left',
      3: 'turn right'
  }

  game = HabitatWrapper(config=config, action_mapping=action_mapping)

  return game

# limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = TEST_MEMORY_FRACTION
set_session(tf.Session(config=config))
