import time
import numpy as np
import tensorflow as tf

import pysc2.lib
import cv2
import joblib
import networks

from tqdm import tqdm
from model import Model
from pysc2.lib import actions
from baselines.a2c.utils import discount_with_dones
from common.vec_env.subproc_vec_env import SubprocVecEnv

class Agent(object):
    def __init__(self, sess, log_path, env_function, num_cpu, network="fullyconv", ar=True, lr=1e-4, optimizer="rmsprop", ent_coef=1e-3, vf_coef=1.0, max_grad_norm=0.5, nsteps=5, nstack=1, gamma=0.99):

        if optimizer == "adam":
            self.trainer = tf.train.AdamOptimizer
        elif optimizer == "rmsprop":
            self.trainer = tf.train.RMSPropOptimizer
        else:
            raise NotImplementedError

        network_func = None
        if network == "fullyconv":
            network_func = networks.FullyConvNet
        elif network == "atari":
            network_func = networks.AtariNet
        else:
            raise NotImplementedError

        self.sess = sess
        self.log_path = log_path
        self.num_cpu = num_cpu
        self.env_function = env_function
        self.init_lr = lr
        self.env = SubprocVecEnv([self.env_function(i) for i in range(1)])
        self.model = Model(network_func=network_func, screen_space=self.env.screen_space, minimap_space=self.env.minimap_space, ns_space=self.env.ns_space,
            trainer=self.trainer, ar=ar, nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
        self.gamma = gamma
        self.nsteps = nsteps
        self.ar = ar

        if ar:
            self.step_func = self.step_policy_ar
        else:
            self.step_func = self.step_policy

    def training_process(self, epoch, train_steps, epsilon):
        # set learning rate
        current_lr = self.init_lr * (0.98**epoch)
        current_lr = max(current_lr, self.init_lr / 2.)

        mb_screen, mb_minimap, mb_ns, mb_rewards, mb_actions, mb_available_actions, mb_pos, mb_values, mb_dones = [],[],[],[],[],[],[],[],[]
        mb_args, mb_args_used = dict(), dict()
        mb_use_spatial_actions = []
        for act_type in actions.TYPES:
            mb_args[act_type.name] = []
            mb_args_used[act_type.name] = []
        mb_states = []

        self.epsilon = epsilon
        self.remake_env(self.num_cpu)
        obs, info = self.env.reset()
        screen, minimap, ns, available_actions = obs["screen"], obs["minimap"], obs["ns"], info["available_actions"]
        states = None
        update_steps = 0
        start_time = time.time()
        print ("=== Training Epoch: ", epoch, ", Learning Rate: ", current_lr , " ===")
        # with tqdm(total=train_steps) as pbar:
        while (True):
            print_log = False
            
            if len(mb_screen) == self.nsteps - 1:
                print_log = True
            
            action, arg, value, state = self.step_func(screen, minimap, ns, states, print_log=print_log, epsilon=epsilon, available_actions=available_actions)
            # action, arg, value, state = self.step_epsilon(screen, minimap, ns, states, print_log=print_log, epsilon=epsilon, available_actions=available_actions)
            mb_screen.append(np.copy(screen))
            mb_minimap.append(np.copy(minimap))
            mb_ns.append(np.copy(ns))
            mb_available_actions.append(np.copy(available_actions))
            mb_actions.append(action)
            # for a in arg:
            for act_type in actions.TYPES:
                temp, temp_used = [], []
                for a in arg:
                    if a[act_type.name] != - 1:
                        temp.append(a[act_type.name])
                        temp_used.append(1.)
                    else:
                        temp.append(0)
                        temp_used.append(0.)
                mb_args[act_type.name].append(temp)
                mb_args_used[act_type.name].append(temp_used)

            mb_values.append(value)
            mb_dones.append(info["last"])
            next_obs, info = self.env.step(action, arg)
            '''
            # This part seems useless. Check later.
            for idx, done in enumerate(info["last"]):
                if done:
                    obs[idx] = obs[idx] * 0
            '''
            obs = next_obs
            mb_rewards.append(info["reward"])
            screen, minimap, ns, available_actions = obs["screen"], obs["minimap"], obs["ns"], info["available_actions"]

            if len(mb_screen) == self.nsteps:
                mb_dones.append(info["last"])
                mb_screen = np.asarray(mb_screen, dtype=np.float32).swapaxes(1, 0).reshape((self.num_cpu * self.nsteps, ) + self.env.screen_space)
                mb_minimap = np.asarray(mb_minimap, dtype=np.float32).swapaxes(1, 0).reshape((self.num_cpu * self.nsteps, ) + self.env.minimap_space)
                mb_ns = np.asarray(mb_ns, dtype=np.float32).swapaxes(1, 0).reshape((self.num_cpu * self.nsteps, ) + self.env.ns_space)
                mb_available_actions = np.asarray(mb_available_actions, dtype=np.float32).swapaxes(1, 0).reshape((self.num_cpu * self.nsteps, ) + (len(pysc2.lib.actions.FUNCTIONS), ))
                mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
                mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
                for act_type in actions.TYPES:
                    mb_args[act_type.name] = np.asarray(mb_args[act_type.name], dtype=np.int32).swapaxes(1, 0)
                    mb_args_used[act_type.name] = np.asarray(mb_args_used[act_type.name], dtype=np.float32).swapaxes(1, 0)
                mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
                mb_dones = np.asarray(mb_dones, dtype=np.float32).swapaxes(1, 0)
                mb_masks = mb_dones[:, :-1]
                mb_dones = mb_dones[:, 1:]
                last_values = self.value(screen, minimap, ns).tolist()
                for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                    rewards = rewards.tolist()
                    dones = dones.tolist()
                    if dones[-1] == 0:
                        rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                    else:
                        rewards = discount_with_dones(rewards, dones, self.gamma)
                    mb_rewards[n] = rewards
                mb_rewards = mb_rewards.flatten()
                mb_actions = mb_actions.flatten()
                for act_type in actions.TYPES:
                    mb_args[act_type.name] = mb_args[act_type.name].flatten()
                    mb_args_used[act_type.name] = mb_args_used[act_type.name].flatten()
                mb_values = mb_values.flatten()
                mb_masks = mb_masks.flatten()
                self.train(current_lr, mb_screen, mb_minimap, mb_ns, mb_states, mb_rewards, mb_masks, mb_actions, mb_use_spatial_actions, mb_available_actions, mb_pos, mb_values, mb_args, mb_args_used)
                update_steps += 1
                # pbar.update(1)
                mb_screen, mb_minimap, mb_ns, mb_rewards, mb_actions, mb_available_actions, mb_pos, mb_values, mb_dones = [],[],[],[],[],[],[],[],[]
                mb_args, mb_args_used = dict(), dict()
                for act_type in actions.TYPES:
                    mb_args[act_type.name] = []
                    mb_args_used[act_type.name] = []

            if update_steps == train_steps:
                break

        self.env.close()
        print("=== Took ", (time.time() - start_time), " seconds to finish ", train_steps , " updates.===")

    def evaluating_process(self, epoch, episodes):
        # Since the game lengths are different for each game, only use one thread when evaluating
        self.remake_env(1)
        rewards = []
        for _ in range(episodes):
            episode_reward = [0]
            obs, info = self.env.reset()
            screen, minimap, ns, available_actions = obs["screen"], obs["minimap"], obs["ns"], info["available_actions"]
            states = None
            while True:
                action, arg, value, state = self.step_policy(screen, minimap, ns, states, available_actions=available_actions)
                obs, info = self.env.step(action, arg)
                episode_reward = [sum(x) for x in zip(episode_reward, info["reward"])]
                screen, minimap, ns, available_actions = obs["screen"], obs["minimap"], obs["ns"], info["available_actions"]
                if info["last"][0]:
                    rewards.append(episode_reward)
                    break
        rewards = [r for sublist in rewards for r in sublist]
        self.env.save_replay("%sreplay" % self.log_path, epoch)
        self.env.close()
        return rewards

    def step_policy_ar(self, screen, minimap, ns, *_args, **_kwargs):
        a_s, a_probs, v_s, args = self.sess.run([self.model.pi_selected, self.model.base_action_softmax, self.model.value, self.model.args_selected],
            {self.model.screen: screen, self.model.minimap: minimap, self.model.ns: ns, self.model.act_mask: _kwargs["available_actions"]})
        
        a_s = np.reshape(a_s, -1)
        a_probs = np.reshape(a_probs, (-1, self.env.base_action_count))

        aid_s, args_s = [], []
        for idx in range(len(a_s)):
            aid = a_s[idx]
            aid_s.append(aid)
            temp_args = dict()
            for k, v in args.items():
                temp_args[k] = -1
            for arg in actions.FUNCTIONS[aid].args:
                temp_args[arg.name] = np.reshape(args[arg.name], -1)[idx]
            args_s.append(temp_args)
        
        if _kwargs.get('print_log', False):
            if args_s[0]["screen"] != -1:
                print ("AR action: ", aid_s[0], " action_prob: ", a_probs[0][aid_s[0]], " pos: (", args_s[0]["screen"] % self.env.screen_space[1], ",", args_s[0]["screen"] // self.env.screen_space[1], ") ", end='')
            else:
                print ("AR action: ", aid_s[0], " action_prob: ", a_probs[0][aid_s[0]], end='')

        return aid_s, args_s, v_s, [] 

    def step_policy(self, screen, minimap, ns, *_args, **_kwargs):
        a_s, v_s, args = self.sess.run([self.model.base_action_softmax, self.model.value, self.model.args],
            {self.model.screen: screen, self.model.minimap: minimap, self.model.ns: ns})

        available_actions = _kwargs["available_actions"]
        filtered_a = np.multiply(a_s, available_actions)
        filtered_a /= np.sum(filtered_a, axis=1, keepdims=True)

        aid_s, args_s = [], []
        for idx in range(np.shape(filtered_a)[0]):
            aid = np.random.choice(len(filtered_a[idx,:]), p=filtered_a[idx,:])
            aid_s.append(aid)
            temp_args = dict()
            # initialize all arguments to -1
            for k, v in args.items():
                temp_args[k] = -1
            # only sample needed arguments
            for arg in actions.FUNCTIONS[aid].args:
                temp_args[arg.name] = np.random.choice(len(args[arg.name][idx]), p=args[arg.name][idx])
            args_s.append(temp_args)

        if _kwargs.get('print_log', False):
            if args_s[0]["screen"] != -1:
                print ("action: ", aid_s[0], " action_prob: ", filtered_a[0][aid_s[0]], " pos: (", args_s[0]["screen"] % self.env.screen_space[1], ",", args_s[0]["screen"] // self.env.screen_space[1], ") pos_prob: ", args["screen"][0][args_s[0]["screen"]], end='')
            else:
                print ("action: ", aid_s[0], " action_prob: ", filtered_a[0][aid_s[0]], end='')

        return aid_s, args_s, v_s, [] 

    '''
    def step_policy(self, screen, minimap, ns, *_args, **_kwargs):
        a_s, v_s, pos_s = self.sess.run([self.model.policy_a, self.model.value, self.model.policy_pos],
            {self.model.screen: screen, self.model.minimap: minimap, self.model.ns: ns, self.model.AVAIL_ACTION: _kwargs["available_actions"]})

        if _kwargs.get('print_log', False):
            print ("action: ", a_s[0], " pos: (", pos_s[0] % 64, ", ", pos_s[0] // 64, ")")
        # print (np.shape(a_s))
        # input()

        return a_s, pos_s, v_s, [] 
    '''

    def step_epsilon(self, screen, minimap, ns, *_args, **_kwargs):
        a_s, v_s, args = self.sess.run([self.model.base_action_softmax, self.model.value, self.model.args],
            {self.model.screen: screen, self.model.minimap: minimap, self.model.ns: ns})

        available_actions = _kwargs["available_actions"]
        filtered_a = np.multiply(a_s, available_actions)
        filtered_a /= np.sum(filtered_a, axis=1, keepdims=True)

        aid_s, args_s = [], []
        for idx in range(np.shape(filtered_a)[0]):
            aid = None
            if np.random.uniform() < self.epsilon:
                available_act_ids = np.nonzero(_kwargs["available_actions"][idx])[0]
                aid = np.random.choice(available_act_ids)
                # print ("Random action:", aid)
            else:
                aid = np.random.choice(len(filtered_a[idx,:]), p=filtered_a[idx,:])
            aid_s.append(aid)
            temp_args = dict()
            # initialize all arguments to -1
            for k, v in args.items():
                temp_args[k] = -1
            # only sample needed arguments
            for arg in actions.FUNCTIONS[aid].args:
                temp_args[arg.name] = np.random.choice(len(args[arg.name][idx]), p=args[arg.name][idx])
            args_s.append(temp_args)

        if _kwargs.get('print_log', False):
            if args_s[0]["screen"] != -1:
                print ("action: ", aid_s[0], " action_prob: ", filtered_a[0][aid_s[0]], " pos: (", args_s[0]["screen"] % self.env.screen_space[1], ",", args_s[0]["screen"] // self.env.screen_space[1], ") pos_prob: ", args["screen"][0][args_s[0]["screen"]], end='')
            else:
                print ("action: ", aid_s[0], " action_prob: ", filtered_a[0][aid_s[0]], end='')

        return aid_s, args_s, v_s, [] 

  
    def train(self, lr, screen, minimap, ns, states, rewards, masks, acts, use_spatial_actions, available_actions, pos, values, args, args_used):
        advs = rewards - values
        td_map = {self.model.screen: screen,
                  self.model.minimap: minimap,
                  self.model.ns: ns,
                  self.model.acts: acts,
                  # self.model.avail_actions: available_actions,
                  self.model.act_mask: available_actions,
                  self.model.advs: advs,
                  self.model.rewards: rewards,
                  self.model.lr: lr}
        for act_type in actions.TYPES:
            td_map[self.model.act_args[act_type.name]] = args[act_type.name]
            td_map[self.model.act_args_used[act_type.name]] = args_used[act_type.name]

        _, pg_loss, neglogpac, entropy, vf_loss = self.sess.run([self.model._train, self.model.pg_loss, self.model.neglogpac, self.model.entropy, self.model.vf_loss], td_map)
        print (" pg_loss: ", pg_loss, " entropy: ", entropy, " vf_loss: ", vf_loss)

    def value(self, screen, minimap, ns, *_args, **_kwargs): 
        v = self.sess.run(self.model.value, {self.model.screen: screen, self.model.minimap: minimap, self.model.ns: ns})
        return v

    def save_model(self, epoch):
        ps = self.sess.run(self.model.params)
        # make_path(save_path)
        open("%s%d.checkpoint" %(self.log_path, epoch), "w+")
        joblib.dump(ps, "%s/%d.checkpoint" %(self.log_path, epoch))

    def remake_env(self, num_cpu):
        self.env.close()
        self.env = SubprocVecEnv([self.env_function(i) for i in range(num_cpu)])