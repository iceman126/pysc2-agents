from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from pysc2.lib import actions
from pysc2.lib import features

import cv2

_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index

_MINIMAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_MINIMAP_SELECTED = features.MINIMAP_FEATURES.selected.index
_MINIMAP_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_MINIMAP_CAMERA_ID = features.MINIMAP_FEATURES.camera.index

def process_catagorical_map(input_map, scale):
    new_map = np.zeros([input_map.shape[0], input_map.shape[1], scale], dtype=np.float32)
    for c in range(scale):
        index_y, index_x = (input_map == c).nonzero()
        new_map[index_y, index_x, c] = 1
    return new_map

def process_state(state):
    screen_features = []
    minimap_features = []
    ns = []

    screen_features.append(process_catagorical_map(state.observation["screen"][_SCREEN_PLAYER_RELATIVE], features.SCREEN_FEATURES[_SCREEN_PLAYER_RELATIVE].scale))
    screen_features.append(process_catagorical_map(state.observation["screen"][_SCREEN_SELECTED], features.SCREEN_FEATURES[_SCREEN_SELECTED].scale))
    screen_features = np.concatenate(screen_features, axis=2)
    # screen_features = screen_features[:,:,[1,3,6]]

    minimap_features.append(process_catagorical_map(state.observation["minimap"][_MINIMAP_PLAYER_RELATIVE], features.MINIMAP_FEATURES[_MINIMAP_PLAYER_RELATIVE].scale))
    minimap_features.append(process_catagorical_map(state.observation["minimap"][_MINIMAP_SELECTED], features.MINIMAP_FEATURES[_MINIMAP_SELECTED].scale))
    minimap_features.append(process_catagorical_map(state.observation["minimap"][_MINIMAP_VISIBILITY], features.MINIMAP_FEATURES[_MINIMAP_VISIBILITY].scale))
    minimap_features.append(process_catagorical_map(state.observation["minimap"][_MINIMAP_CAMERA_ID], features.MINIMAP_FEATURES[_MINIMAP_CAMERA_ID].scale))
    minimap_features = np.concatenate(minimap_features, axis=2)
    
    # action list: https://github.com/deepmind/pysc2/blob/master/pysc2/lib/actions.py
    forbid_actions = [1, 13, 3, 4, 332, 333, 334, 452]

    # ns = state.observation["single_select"][:, 1]
    # print (ns.shape)
    # input('======')

    
    ns = np.zeros([len(actions.FUNCTIONS)], dtype=np.float32)
    ns[state.observation["available_actions"]] = 1
    ns[forbid_actions] = 0
    

    available_actions = np.zeros([len(actions.FUNCTIONS)], dtype=np.float32)
    available_actions[state.observation["available_actions"]] = 1
    available_actions[forbid_actions] = 0

    ob = {"screen": screen_features, "minimap": minimap_features, "ns": ns}
    info = {"available_actions": available_actions}

    return ob, info

def copy_graph(to_scope, from_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def mse(pred, target):
    return tf.square(pred - target) / 2.