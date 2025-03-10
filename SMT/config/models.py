from yacs.config import CfgNode as CN

from . import default

HIGH_RESOLUTION_NET = CN()

SCENE_MEMORY = CN()
SCENE_MEMORY.MODUALITIES = ['rgb', 'pose', 'prev_action']
SCENE_MEMORY.MODUALITY_DIM = [64, 16,16]
SCENE_MEMORY.REDUCE_FACTOR = 4

ATTENTION_POLICY = CN()
ATTENTION_POLICY.NUM_CLASSES = default._C.TASK.NUM_ACTIONS
ATTENTION_POLICY.DEPTH = 128
ATTENTION_POLICY.NUM_HEADS = 8
ATTENTION_POLICY.D_MODEL = ATTENTION_POLICY.DEPTH*ATTENTION_POLICY.NUM_HEADS
ATTENTION_POLICY.EPSILON = 1e-6
ATTENTION_POLICY.RATE = 0.1
