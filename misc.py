import os
import numpy as np
import random

from pysc2.lib import actions

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def is_spatial_action(act_ids):
    result = []
    for act_id in act_ids:
        args = actions.FUNCTIONS[act_id].args
        is_spatial = False
        for arg in args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                is_spatial = True
                break
        if is_spatial:
            result.append(1)
        else:
            result.append(0)
    return result

def make_path(f):
    return os.makedirs(f, exist_ok=True)