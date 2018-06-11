import numpy as np
import tensorflow as tf
from utils import mse
from pysc2.lib import actions
from tensorflow.contrib.distributions import Categorical

class Model(object):
    def __init__(self, network_func, screen_space, minimap_space, ns_space, nstack, trainer, ar,
            ent_coef=0.00001, vf_coef=1.0, max_grad_norm=None):
        nact = len(actions.FUNCTIONS)
        self.screen, self.minimap, self.ns, self.act_mask, self.base_action_softmax, self.value_layer, self.args, self.pi_selected, self.args_selected =\
              network_func("model", screen_space, minimap_space, ns_space, nact, ar)

        self.value = self.value_layer[:, 0]
        acts = tf.placeholder(tf.int32, [None])
        act_args = {act_type.name: tf.placeholder(tf.int32, [None]) for act_type in actions.TYPES}
        act_args_used = {act_type.name: tf.placeholder(tf.float32, [None]) for act_type in actions.TYPES}
        advs = tf.placeholder(tf.float32, [None])
        rewards = tf.placeholder(tf.float32, [None])
        lr = tf.placeholder(tf.float32, [])

        acts_one_hot = tf.one_hot(acts, nact)

        # action log probability
        valid_action_prob_sum = tf.reduce_sum(self.base_action_softmax * self.act_mask, axis=1)       # sum all valid non spatial action prob
        action_prob = tf.reduce_sum(self.base_action_softmax * acts_one_hot, axis=1)
        action_log_prob = tf.log(action_prob / valid_action_prob_sum + 1e-8)

        neglogpac = action_log_prob
    
        for act_type in actions.TYPES:
            indexes = tf.stack([tf.range(tf.shape(act_args[act_type.name])[0]), act_args[act_type.name]], axis=1)
            arg_log_prob = tf.log(tf.gather_nd(self.args[act_type.name], indexes) + 1e-8)
            neglogpac += act_args_used[act_type.name] * arg_log_prob

        action_entropy = tf.reduce_mean(-tf.reduce_sum(self.base_action_softmax * tf.log(self.base_action_softmax + 1e-8), axis=1))

        # args entropy
        entropy = action_entropy
        for act_type in actions.TYPES:
            entropy += tf.reduce_sum(-tf.reduce_sum(self.args[act_type.name] * tf.log(self.args[act_type.name] + 1e-8), axis=1) * act_args_used[act_type.name]) / tf.maximum(tf.reduce_sum(act_args_used[act_type.name]), 1.)

        pg_loss = -tf.reduce_mean(advs * neglogpac)
        vf_loss = tf.reduce_mean(tf.square(self.value - rewards) / 2.)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        opt_trainer = trainer(learning_rate=lr)
        _train = opt_trainer.apply_gradients(grads)

        self.lr = lr
        self.acts = acts
        self.act_args = act_args
        self.act_args_used = act_args_used
        # self.avail_actions = avail_actions
        self.advs = advs
        self.rewards = rewards
        self._train = _train
        self.policy_loss = pg_loss
        self.value_loss = vf_loss
        self.entropy = entropy
        self.params = params
        self.pg_loss = pg_loss
        self.entropy = entropy
        self.vf_loss = vf_loss
        self.neglogpac = neglogpac
        # self.base_action_softmax = tf.nn.softmax(self.base_action)
        # self.args_softmax = {act_type.name: tf.nn.softmax(self.args[act_type.name]) for act_type in actions.TYPES}

        # self.policy_a = self.sample_from_dist(self.base_action_softmax * self.AVAIL_ACTION / tf.reduce_sum(self.base_action_softmax * self.AVAIL_ACTION, axis=1, keepdims=True))
        # self.policy_pos = self.sample_from_dist(self.action_pos_softmax)

def load(load_path):
    loaded_params = joblib.load(load_path)
    restores = []
    for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
    ps = sess.run(restores)

def print_params():
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print ("Variable: ", k)
        print ("Shape: ", v.shape)
        # print v
    # ps = sess.run(params)
    # print (ps)