import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,1)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.LOSS = CN()

_C.MODEL = CN()
_C.MODEL.NAME = 'smt'

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'gibson'
_C.DATASET.TRAIN_SET = ''
_C.DATASET.TEST_SET = ''
#_C.DATASET.ACTION_MAPPING = {0: 'move_forward', 1: 'turn left', 2: 'turn right', 3: 'stop'}

_C.TASK = CN()
_C.TASK.NAME = 'coverage'
_C.TASK.MODUALITIES = ['image']
_C.TASK.CELL_HEIGHT = 0.5
_C.TASK.CELL_WIDTH = 0.5
_C.TASK.REWARD_RATE = 5
_C.TASK.NUM_ACTIONS = 4
_C.TASK.ACTION_NAMES = ['move_forward', 'turn left', 'turn right', 'stop']
_C.TASK.RANDOM_EPISODES_THRESHOLD = 1000
_C.TASK.ALIGH_MODEL_THRESHOLD = 20
_C.TASK.HORIZON = 100

# training
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LR = 5e-4
_C.TRAIN.NUM_ITERATIONS = 100
_C.TRAIN.EPISODES_PER_SCENE = 20

# testing
_C.TEST = CN()

# loss
_C.LOSS = CN()
_C.LOSS.TYPE = 'mse'

def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

