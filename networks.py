import numpy as np
import tensorflow as tf
# from tensorflow.distributions import Categorical

from pysc2.lib import features
from pysc2.lib import actions

def AtariNet(scope, screen_space, minimap_space, ns_space, nact, ar=True):
    screen = tf.placeholder(tf.float32, ((None,) + screen_space))
    minimap = tf.placeholder(tf.float32, ((None,) + minimap_space))
    ns = tf.placeholder(tf.float32, ((None,) + ns_space))
    act_mask = tf.placeholder(tf.float32, (None, nact))
    with tf.variable_scope("model"):
        screen_processed = embed_features(screen, features.SCREEN_FEATURES)
        minimap_processed = embed_features(minimap, features.MINIMAP_FEATURES)
        screen_conv1 = tf.layers.conv2d(
                screen_processed,
                filters=16,
                kernel_size=8,
                strides=4,
                # kernel_initializer=tf.glorot_uniform_initializer(),
                kernel_initializer=tf.orthogonal_initializer(),
                # kernel_initializer=tf.glorot_normal_initializer(),
                name="screen_conv1",
                padding="same",
                activation=tf.nn.relu
        )
        screen_conv2 = tf.layers.conv2d(
                screen_conv1,
                filters=32,
                kernel_size=4,
                strides=2,
                # kernel_initializer=tf.glorot_uniform_initializer(),
                kernel_initializer=tf.orthogonal_initializer(),
                # kernel_initializer=tf.glorot_normal_initializer(),
                name="screen_conv2",
                padding="same",
                activation=tf.nn.relu
        )
        minimap_conv1 = tf.layers.conv2d(
                minimap_processed,
                filters=16,
                kernel_size=8,
                strides=4,
                # kernel_initializer=tf.glorot_uniform_initializer(),
                kernel_initializer=tf.orthogonal_initializer(),
                # kernel_initializer=tf.glorot_normal_initializer(),
                name="minimap_conv1",
                padding="same",
                activation=tf.nn.relu
        )
        minimap_conv2 = tf.layers.conv2d(
                minimap_conv1,
                filters=32,
                kernel_size=4,
                strides=2,
                # kernel_initializer=tf.glorot_uniform_initializer(),
                kernel_initializer=tf.orthogonal_initializer(),
                # kernel_initializer=tf.glorot_normal_initializer(),
                name="minimap_conv2",
                padding="same",
                activation=tf.nn.relu
        )
        
        screen_flatten = tf.layers.flatten(screen_conv2, name="screen_flatten")
        minimap_flatten = tf.layers.flatten(minimap_conv2, name="minimap_flatten")

        ns_fc = tf.layers.dense(ns, 256,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.orthogonal_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            activation=tf.nn.tanh,
            name="ns_fc"
        )

        fc_concat = tf.concat([screen_flatten, minimap_flatten, ns_fc], axis=1)
        # fc_concat = tf.concat([screen_flatten, ns_fc], axis=1)

        state_representation = tf.layers.dense(fc_concat, 256,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.orthogonal_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            activation=tf.nn.relu,
            name="state_representation"
        )

        pi = tf.layers.dense(state_representation, nact,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.orthogonal_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            activation=tf.nn.softmax,
            name="pi"
        ) 

        vf = tf.layers.dense(state_representation, 1,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.orthogonal_initializer(),
            # kernel_initializer=tf.glorot_normal_initializer(),
            activation=None,
            name="vf"
        )

        if ar:
            pi_masked = pi * act_mask
            pi_dist = tf.distributions.Categorical(probs=pi_masked / tf.reduce_sum(pi_masked, axis=1, keepdims=True))
            pi_selected = pi_dist.sample()
            pi_one_hot = tf.stop_gradient(tf.one_hot(pi_selected, nact))

            args_output = dict()
            args_selected = dict()
            args_one_hot = dict()

            first_layer_args = ['queued', 'select_add', 'select_worker', 'build_queue_id', 'control_group_act', 'select_unit_act', 'unload_id', 'select_point_act']
            # first layer arguments
            for act_type in actions.TYPES:
                if act_type.name in first_layer_args:
                    arg_out = non_spatial_argument(state_representation, act_type.sizes[0], act_type.name)
                    args_output[act_type.name] = arg_out
                    args_selected[act_type.name] = tf.distributions.Categorical(probs=arg_out).sample()
                    args_one_hot[act_type.name] = tf.stop_gradient(tf.one_hot(args_selected[act_type.name], act_type.sizes[0]))

            # second layer arguments
            ## control group id
            ctrl_group_id_input = tf.concat([state_representation, pi_one_hot, args_one_hot["control_group_act"]], axis=1)
            args_output["control_group_id"] = non_spatial_argument(ctrl_group_id_input, actions.TYPES[5].sizes[0], "control_group_id")
            args_selected["control_group_id"] = tf.distributions.Categorical(probs=args_output["control_group_id"]).sample()
            args_one_hot["control_group_id"] = tf.stop_gradient(tf.one_hot(args_selected["control_group_id"], actions.TYPES[5].sizes[0]))

            ## select unit id
            select_unit_id_input = tf.concat([state_representation, pi_one_hot, args_one_hot["select_unit_act"]], axis=1)
            args_output["select_unit_id"] = non_spatial_argument(select_unit_id_input, actions.TYPES[9].sizes[0],"select_unit_id")
            args_selected["select_unit_id"] = tf.distributions.Categorical(probs=args_output["select_unit_id"]).sample()
            args_one_hot["select_unit_id"] = tf.stop_gradient(tf.one_hot(args_selected["select_unit_id"], actions.TYPES[9].sizes[0]))

            ## screen, screen2
            for name in ('screen', 'screen2'):
                concat_input = tf.concat([state_representation, pi_one_hot, args_one_hot["queued"], args_one_hot["select_point_act"], args_one_hot["select_add"]], axis=1)
                args_output[name] = fc_spatial_argument(concat_input, screen_space[0], screen_space[1], name)
                args_selected[name] = tf.distributions.Categorical(probs=args_output[name]).sample()
                args_one_hot[name] = tf.stop_gradient(tf.one_hot(args_selected[name], screen_space[0] * screen_space[1]))

            # minimap
            minimap_input = tf.concat([state_representation, pi_one_hot, args_one_hot["queued"]], axis=1)
            args_output["minimap"] = fc_spatial_argument(minimap_input, minimap_space[0], minimap_space[1], "minimap")
            args_selected["minimap"] = tf.distributions.Categorical(probs=args_output["minimap"]).sample()
            args_one_hot["minimap"] = tf.stop_gradient(tf.one_hot(args_selected["minimap"], minimap_space[0] * minimap_space[1]))

            return screen, minimap, ns, act_mask, pi, vf, args_output, pi_selected, args_selected

        else:
            args_output = dict()
            for act_type in actions.TYPES:
                if act_type.name in ("screen", "screen2"):
                    arg_out = fc_spatial_argument(state_representation, screen_space[0], screen_space[1], act_type.name)
                elif act_type.name == "minimap":
                    arg_out = fc_spatial_argument(state_representation, minimap_space[0], minimap_space[1], act_type.name)
                else:
                    arg_out = non_spatial_argument(state_representation, act_type.sizes[0], act_type.name)
                args_output[act_type.name] = arg_out

            return screen, minimap, ns, act_mask, pi, vf, args_output, [], []



def fc_spatial_argument(state, height, width, name):
    x = tf.layers.dense(state, width,
        # kernel_initializer=tf.glorot_uniform_initializer(),
        kernel_initializer=tf.orthogonal_initializer(),
        # kernel_initializer=tf.glorot_normal_initializer(),
        activation=tf.nn.softmax,
        name="%s_x" % name
    )
    y = tf.layers.dense(state, height,
        # kernel_initializer=tf.glorot_uniform_initializer(),
        kernel_initializer=tf.orthogonal_initializer(),
        # kernel_initializer=tf.glorot_normal_initializer(),
        activation=tf.nn.softmax,
        name="%s_y" % name
    )
    pos = tf.layers.flatten(tf.reshape(x, [-1, 1, height]) * tf.reshape(y, [-1, width, 1]))
    return pos

def embed_features(t, input_features):
    split_features = tf.split(t, len(input_features), -1)   # split the data along last dimension (channel)
    out = None
    map_list = []
    for idx, feature in enumerate(input_features):
        if feature.type == features.FeatureType.CATEGORICAL:
            dims = np.round(np.log2(feature.scale)).astype(np.int32).item()
            dims = max(dims, 1)
            one_hot_maps = tf.one_hot(tf.to_int32(tf.squeeze(split_features[idx], -1)), feature.scale)
            out = tf.layers.conv2d(one_hot_maps,
                filters=dims,
                kernel_size=1,
                strides=1,
                # kernel_initializer=tf.glorot_uniform_initializer(),
                # kernel_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.orthogonal_initializer(),
                padding="same",
                activation=tf.nn.relu
            )
        elif feature.type == features.FeatureType.SCALAR: 
            out = tf.log1p(split_features[idx])
        else:
            raise AttributeError
        map_list.append(out)
    processed_features = tf.concat(map_list, -1)
    return processed_features

def FullyConvNet(scope, screen_space, minimap_space, ns_space, nact):
    screen = tf.placeholder(tf.float32, ((None,) + screen_space) )
    minimap = tf.placeholder(tf.float32, ((None,) + minimap_space) )
    ns = tf.placeholder(tf.float32, ((None,) + ns_space))
    with tf.variable_scope("model"):
        screen_processed = embed_features(screen, features.SCREEN_FEATURES)
        minimap_processed = embed_features(minimap, features.MINIMAP_FEATURES)

        screen_conv1 = tf.layers.conv2d(
            screen_processed,
            filters=16,
            kernel_size=5,
            strides=1,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            name="screen_conv1",
            padding="same",
            activation=tf.nn.relu
        )
        screen_conv2 = tf.layers.conv2d(
            screen_conv1,
            filters=32,
            kernel_size=3,
            strides=1,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            name="screen_conv2",
            padding="same",
            activation=tf.nn.relu
        )
        
        minimap_conv1 = tf.layers.conv2d(
            minimap_processed,
            filters=16,
            kernel_size=5,
            strides=1,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            name="minimap_conv1",
            padding="same",
            activation=tf.nn.relu
        )
        minimap_conv2 = tf.layers.conv2d(
            minimap_conv1,
            filters=32,
            kernel_size=3,
            strides=1,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            name="minimap_conv2",
            padding="same",
            activation=tf.nn.relu
        )
        

        ns_broadcast = tf.tile( tf.expand_dims(tf.expand_dims(ns, 1), 2), tf.stack([1, screen_space[0], screen_space[1], 1]) )

        state_representation = tf.concat([screen_conv2, minimap_conv2, ns_broadcast], axis=3)
        # state_representation = tf.concat([screen_conv2, ns_broadcast], axis=3, name="state_representation")
        state_representation_flattened = tf.layers.flatten(state_representation, name="state_representation_flattened")

        fc1 = tf.layers.dense(state_representation_flattened, 256,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=tf.nn.relu,
            name="fc1"
        )

        pi = tf.layers.dense(fc1, nact,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=tf.nn.softmax,
            # activation=None,
            name="pi"
        )

        vf = tf.layers.dense(fc1, 1,
            # kernel_initializer=tf.glorot_uniform_initializer(),
            kernel_initializer=tf.glorot_normal_initializer(),
            # kernel_initializer=tf.orthogonal_initializer(),
            activation=None,
            name="vf"
        )

        args_output = dict()
        args_output_log = dict()
        for act_type in actions.TYPES:
            arg_out = None
            # arg_out_log = None
            if len(act_type.sizes) > 1:         # spatial action argument
                arg_out = spatial_argument(state_representation, act_type.name)
            else:
                arg_out = non_spatial_argument(fc1, act_type.sizes[0], act_type.name)
            args_output[act_type.name] = arg_out
            # args_output_log[act_type.name] = arg_out_log

    return screen, minimap, ns, pi, vf, args_output

def spatial_argument(state, name):
    temp = tf.layers.conv2d(
        state,
        filters=1,
        kernel_size=1,
        strides=1,
        # kernel_initializer=tf.glorot_uniform_initializer(),
        kernel_initializer=tf.glorot_normal_initializer(),
        # kernel_initializer=tf.orthogonal_initializer(),
        activation=None,
        padding="same",
        name="arg_%s" % name
    )
    return tf.nn.softmax(tf.layers.flatten(temp))
    
def non_spatial_argument(fc, size, name):
    temp = tf.layers.dense(fc, size,
        # kernel_initializer=tf.glorot_uniform_initializer(),
        # kernel_initializer=tf.glorot_normal_initializer(),
        kernel_initializer=tf.orthogonal_initializer(),
        activation=tf.nn.softmax,
        # activation=None,
        name="arg_%s" % name
    )
    return temp