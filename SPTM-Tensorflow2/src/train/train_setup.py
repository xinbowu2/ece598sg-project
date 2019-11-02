import sys
#sys.path.append('../')
# limit memory usage
#sys.path.append('common')
from common import *
import tensorflow as tf

def setup_training_paths(experiment_id):
  experiment_path = EXPERIMENTS_PATH_TEMPLATE % experiment_id
  logs_path = LOGS_PATH_TEMPLATE % experiment_id
  models_path = MODELS_PATH_TEMPLATE % experiment_id
  current_model_path = CURRENT_MODEL_PATH_TEMPLATE % experiment_id
  assert (not os.path.exists(experiment_path)), 'Experiment folder %s already exists' % experiment_path
  os.makedirs(experiment_path)
  os.makedirs(logs_path)
  os.makedirs(models_path)
  return logs_path, current_model_path
