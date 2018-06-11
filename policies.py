import numpy as np
import tensorflow as tf
# import tensorflow.contrib.layers as contrib_layers
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

'''
def select_action(x, available_actions):
    result = []
    for env_idx in range(len(x)):
        act_idx = np.nonzero(available_actions[env_idx])[0]
        # print (act_idx)
        act_values = x[env_idx][act_idx]
        act_softmax = softmax(act_values)
        random_num = np.random.uniform()
        for i in range(len(act_values)):
            random_num -= act_softmax[i]
            if random_num <= 0:
                result.append(act_idx[i])
                break
    return np.array(result)
'''

def select_action_greedy(x, available_actions):
    result = []
    result_prob = []
    for env_idx in range(len(x)):
        act_idx = np.nonzero(available_actions[env_idx])[0]
        act_softmax = x[env_idx][act_idx]
        act_prob = act_softmax / np.sum(act_softmax)
        result.append(act_idx[np.argmax(act_prob)])
        result_prob.append(act_prob[np.argmax(act_prob)])
    return np.array(result), np.array(result_prob)

def select_action_policy(x, available_actions):
    result = []
    result_prob = []
    for env_idx in range(len(x)):
        act_idx = np.nonzero(available_actions[env_idx])[0]
        act_softmax = x[env_idx][act_idx]
        act_prob = act_softmax / np.sum(act_softmax)

        random_num = np.random.uniform()
        for i in range(len(act_prob)):
            random_num -= act_prob[i]
            if random_num <= 0:
                result.append(act_idx[i])
                result_prob.append(act_prob[i])
                break
    return np.array(result), np.array(result_prob)

'''
def select_position(x):
    result = []
    for env_idx in range(len(x)):
        pos_softmax = softmax(x[env_idx])
        random_num = np.random.uniform()
        for i in range(len(pos_softmax)):
            random_num -= pos_softmax[i]
            if random_num <= 0:
                result.append(i)
                break
    return np.array(result)
'''

def select_position_policy(x):
    result = []
    for env_idx in range(len(x)):
        pos_softmax = x[env_idx]
        random_num = np.random.uniform()
        for i in range(len(pos_softmax)):
            random_num -= pos_softmax[i]
            if random_num <= 0:
                result.append(i)
                break
    return np.array(result)

def softmax_sample(x):
    result = []
    random_num = np.random.uniform()
    for env_idx in range(len(x)):
        for i in range(len(x[env_idx])):
            random_num -= x[env_idx][i]
            if random_num <= 0:
                result.append(i)
                break
    return np.array(result)

class Atari(object):
    def __init__(self, sess, screen_space, minimap_space, ns_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        screen_nh, screen_nw, screen_nc = screen_space

        minimap_nh, minimap_nw, minimap_nc = minimap_space
        ns_len = ns_space
        nact = ac_space

        screen_shape = (None, screen_nh, screen_nw, screen_nc*nstack)
        minimap_shape = (None, minimap_nh, minimap_nw, minimap_nc*nstack)
        ns_shape = (None, ns_len)        # non spatial features

        screen = tf.placeholder(tf.float32, screen_shape)
        minimap = tf.placeholder(tf.float32, minimap_shape)
        ns = tf.placeholder(tf.float32, ns_shape)

        with tf.variable_scope("model", reuse=reuse):
            screen_conv1 = tf.layers.conv2d(
                    screen,
                    filters=16,
                    kernel_size=8,
                    strides=4,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name="screen_conv1",
                    padding="SAME",
                    activation=tf.nn.relu
            )
            screen_conv2 = tf.layers.conv2d(
                    screen_conv1,
                    filters=32,
                    kernel_size=4,
                    strides=2,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name="screen_conv2",
                    padding="SAME",
                    activation=tf.nn.relu
            )

            minimap_conv1 = tf.layers.conv2d(
                    minimap,
                    filters=16,
                    kernel_size=8,
                    strides=4,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name="minimap_conv1",
                    padding="SAME",
                    activation=tf.nn.relu
            )
            minimap_conv2 = tf.layers.conv2d(
                    minimap_conv1,
                    filters=32,
                    kernel_size=4,
                    strides=2,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name="minimap_conv2",
                    padding="SAME",
                    activation=tf.nn.relu
            )

            screen_flatten = tf.layers.flatten(screen_conv2, name="screen_flatten")
            minimap_flatten = tf.layers.flatten(minimap_conv2, name="minimap_flatten")

            ns_fc = tf.layers.dense(ns, 256,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    activation=tf.nn.tanh,
                    name="ns_fc"
            )

            fc_concat = tf.concat([screen_flatten, minimap_flatten, ns_fc], axis=1)

            state_representation = tf.layers.dense(fc_concat, 256,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    activation=tf.nn.relu,
                    name="state_representation"
            )

            action_x = tf.layers.dense(state_representation, 64,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    activation=None,
                    name="action_x"
            )
            action_y = tf.layers.dense(state_representation, 64,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    activation=None,
                    name="action_y"
            )

            '''
            spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
            spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
            spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
            spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
            spatial_action = layers.flatten(spatial_action_x * spatial_action_y)
            '''

            action_pos = tf.layers.flatten(tf.reshape(action_x, [-1, 1, 64]) * tf.reshape(action_y, [-1, 64, 1]))
            # action_pos = tf.cross(action_x, action_y)

            pi = tf.layers.dense(state_representation, nact,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    activation=None,
                    name="pi"
            ) 

            vf = tf.layers.dense(state_representation, 1,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    activation=None,
                    name="vf"
            )

        '''
        with tf.variable_scope("model", reuse=reuse):
            screen_conv1 = conv(tf.cast(screen, tf.float32), 'sc1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2))
            screen_conv2 = conv(screen_conv1, 'sc2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))
            screen_flatten = conv_to_fc(screen_conv2)
            # minimap_conv1 = conv(tf.cast(minimap, tf.float32), 'mc1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2))
            # minimap_conv2 = conv(minimap_conv1, 'mc2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))
            # minimap_flatten = conv_to_fc(minimap_conv2)

            ns_fc = fc(tf.cast(ns, tf.float32), 'nfc', nh=64, act=tf.nn.tanh)

            fc_concat = tf.concat(axis=1, values=[screen_flatten, ns_fc])
            state_representation = fc(fc_concat, 'state', nh=256)

            action_x = fc(state_representation, 'action_x', nh=screen_nw, act=tf.nn.softmax)
            action_y = fc(state_representation, 'action_y', nh=screen_nh, act=tf.nn.softmax)

            pi = fc(state_representation, 'action', nact, act=tf.nn.softmax)
            vf = fc(state_representation, 'v', 1, act=lambda x:x)
        '''

        v0 = vf[:, 0]
        a0 = pi
        a0_softmax = tf.nn.softmax(pi)
        a_x = action_x
        a_y = action_y
        a_pos = action_pos
        a_x_softmax = tf.nn.softmax(action_x)
        a_y_softmax = tf.nn.softmax(action_y)
        a_pos_softmax = tf.nn.softmax(action_pos)
        self.initial_state = [] # not stateful

        def step(screen, minimap, ns, *_args, **_kwargs):
            # decide x y together
            a, v, pos_softmax = sess.run([a0_softmax, v0, a_pos_softmax], {self.screen: screen, self.minimap: minimap, self.ns: ns})
            a, a_prob = select_action(a, _kwargs["available_actions"])
            pos = np.argmax(pos_softmax, axis=1)
            return a, pos, v, [], a_prob


            '''
            # x y are independent
            a, v, x_softmax, y_softmax = sess.run([a0_softmax, v0, a_x_softmax, a_y_softmax], {self.screen: screen, self.minimap: minimap, self.ns: ns})
            a, a_prob = select_action(a, _kwargs["available_actions"])
            pos_x = select_position(x_softmax)
            pos_y = select_position(y_softmax)
            return a, pos_x, pos_y, v, [], a_prob # dummy state
            '''


        def step_policy(screen, minimap, ns, *_args, **_kwargs):
            a, v, pos_softmax = sess.run([a0_softmax, v0, a_pos_softmax], {self.screen: screen, self.minimap: minimap, self.ns: ns})
            a, a_prob = select_action_policy(a, _kwargs["available_actions"])
            pos = select_position_policy(pos_softmax)
            print_log = _kwargs.get('print_log', False)
            if print_log == True:
                print ("actions: ", a, "action prob:", a_prob, " position: ", np.asarray(pos) % 64, np.asarray(pos) // 64, "values: ", v)
            return a, pos, v, []

        def step_greedy(screen, minimap, ns, epsilon, *_args, **_kwargs):
            a, v, pos_softmax = sess.run([a0_softmax, v0, a_pos_softmax], {self.screen: screen, self.minimap: minimap, self.ns: ns})
            a, a_prob = select_action_greedy(a, _kwargs["available_actions"])
            pos = np.argmax(pos_softmax, axis=1)
            print_log = _kwargs.get('print_log', False)
            if print_log == True:
                print ("actions: ", a, "action prob:", a_prob, " position: ", np.asarray(pos) % 64, np.asarray(pos) // 64, "values: ", v)
            return a, pos, v, []

        def step_epsilon(screen, minimap, ns, *_args, **_kwargs):
            a, v, pos_softmax = sess.run([a0_softmax, v0, a_pos_softmax], {self.screen: screen, self.minimap: minimap, self.ns: ns})
            a, a_prob = select_action_greedy(a, _kwargs["available_actions"])
            pos = np.argmax(pos_softmax, axis=1)
            epsilon = _kwargs["epsilon"]

            actions = []
            for env_idx in range(a.shape[0]):
                # print (self.available_actions[env_idx])
                if np.random.uniform() < epsilon:       # random_action
                    available_act_ids = np.nonzero(_kwargs["available_actions"][env_idx])[0]
                    selected_id = np.random.choice(available_act_ids)
                    actions.append(selected_id)
                else:
                    actions.append(a[env_idx])

            positions = []
            for env_idx in range(a.shape[0]):
                # print (self.available_actions[env_idx])
                if np.random.uniform() < epsilon:       # random_action

                    x_low = (pos[env_idx] % 64) - 15
                    if x_low < 0:
                        x_low = 0
                    x_high = (pos[env_idx] % 64) + 15
                    if x_high > 63:
                        x_high = 63

                    y_low = (pos[env_idx] // 64) - 15
                    if y_low < 0:
                        y_low = 0
                    y_high = (pos[env_idx] // 64) + 15
                    if y_high > 63:
                        y_high = 63

                    temp_x = np.random.random_integers(low=x_low, high=x_high)
                    temp_y = np.random.random_integers(low=y_low, high=y_high)
                    positions.append(temp_y * 64 + temp_x)
                else:
                    positions.append(pos[env_idx])

            print_log = _kwargs.get('print_log', False)
            if print_log == True:
                print ("policy actions: ", a, "action prob:", a_prob, "epsilon:", epsilon ," applied_actions: ", actions, " position: ", np.asarray(positions) % 64, np.asarray(positions) // 64, "values: ", v)

            return actions, positions, v, []

        def value(screen, minimap, ns, *_args, **_kwargs):
            return sess.run(v0, {self.screen: screen, self.minimap: minimap, self.ns: ns})

        #self.X = X
        self.screen = screen
        self.minimap = minimap
        self.ns = ns
        self.pi = a0
        self.pi_softmax = a0_softmax
        # self.spatial_pi = spatial_pi
        self.pi_x = action_x
        self.pi_y = action_y
        self.pi_pos = action_pos
        self.pi_x_softmax = a_x_softmax
        self.pi_y_softmax = a_y_softmax
        self.pi_pos_softmax = a_pos_softmax
        self.vf = vf
        self.step = step
        self.step_epsilon = step_epsilon
        self.step_greedy = step_greedy
        self.step_policy = step_policy
        self.value = value

'''
class Atari(object):
    def __init__(self, sess, screen_space, minimap_space, ns_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        screen_nh, screen_nw, screen_nc = screen_space

        minimap_nh, minimap_nw, minimap_nc = minimap_space
        ns_len = ns_space
        nact = ac_space

        screen_shape = (nbatch, screen_nh, screen_nw, screen_nc*nstack)
        minimap_shape = (nbatch, minimap_nh, minimap_nw, minimap_nc*nstack)
        ns_shape = (nbatch, ns_len)        # non spatial features

        screen = tf.placeholder(tf.uint8, screen_shape)
        minimap = tf.placeholder(tf.uint8, minimap_shape)
        ns = tf.placeholder(tf.uint8, ns_shape)

        with tf.variable_scope("model", reuse=reuse):
            screen_conv1 = conv(tf.cast(screen, tf.float32), 'sc1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2))
            screen_conv2 = conv(screen_conv1, 'sc2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))
            screen_flatten = conv_to_fc(screen_conv2)
            minimap_conv1 = conv(tf.cast(minimap, tf.float32), 'mc1', nf=16, rf=8, stride=4, init_scale=np.sqrt(2))
            minimap_conv2 = conv(minimap_conv1, 'mc2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))
            minimap_flatten = conv_to_fc(minimap_conv2)

            ns_fc = fc(tf.cast(ns, tf.float32), 'nfc', nh=64, act=tf.nn.tanh)

            fc_concat = tf.concat(axis=1, values=[screen_flatten, minimap_flatten, ns_fc])
            state_representation = fc(fc_concat, 'state', nh=256)

            action_x = fc(state_representation, 'action_x', nh=screen_nw, act=tf.nn.softmax)
            action_y = fc(state_representation, 'action_y', nh=screen_nh, act=tf.nn.softmax)

            pi = fc(state_representation, 'action', nact, act=tf.nn.softmax)
            vf = fc(state_representation, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = pi
        a_x = action_x
        a_y = action_y
        self.initial_state = [] # not stateful

        def step(screen, minimap, ns, *_args, **_kwargs):
            # print (_kwargs["infos"])
            # input('=========')
            a, v, x_softmax, y_softmax = sess.run([a0, v0, a_x, a_y], {self.screen: screen, self.minimap: minimap, self.ns: ns})
            a = select_action(a, _kwargs["available_actions"])
            pos_x = select_position(x_softmax)
            pos_y = select_position(y_softmax)
            return a, pos_x, pos_y, v, [] # dummy state

        def value(screen, minimap, ns, *_args, **_kwargs):
            return sess.run(v0, {self.screen: screen, self.minimap: minimap, self.ns: ns})

        #self.X = X
        self.screen = screen
        self.minimap = minimap
        self.ns = ns
        self.pi = pi
        # self.spatial_pi = spatial_pi
        self.pi_x = action_x
        self.pi_y = action_y
        self.vf = vf
        self.step = step
        self.value = value
'''