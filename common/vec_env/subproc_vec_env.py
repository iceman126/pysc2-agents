import os
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv

# from utils import process_state
from pysc2.env import environment
from pysc2.lib import actions
from pysc2.lib import features

_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index

_MINIMAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_MINIMAP_SELECTED = features.MINIMAP_FEATURES.selected.index
_MINIMAP_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_MINIMAP_CAMERA_ID = features.MINIMAP_FEATURES.camera.index

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, info = process_state(env.step(data)[0])                # pysc2 uses the name "timestep" for current observation
            remote.send((ob, info))

        elif cmd == 'reset':
            ob, info = process_state(env.reset()[0])
            remote.send((ob, info))

        elif cmd == 'close':
            remote.close()
            break

        elif cmd == 'save_replay':
            path, epoch, env_idx = data["path"], data["epoch"], data["env_idx"]
            replay_filename = "%d_%d.SC2Replay" % (epoch, env_idx)
            if not os.path.exists(path):
                os.makedirs(path)
            replay_path = os.path.join(path, replay_filename)
            with open(replay_path, "wb") as f:
                f.write(env._controllers[0].save_replay())

        elif cmd == 'get_spaces':
            '''
            screen_spec = env.observation_spec()["feature_screen"]
            minimap_spec = env.observation_spec()["feature_minimap"]
            # remote.send(((screen_spec[1], screen_spec[2], screen_spec[0]), (minimap_spec[1], minimap_spec[2], minimap_spec[0]), env.observation_spec()["player"], (len(actions.FUNCTIONS),)))
            remote.send(((screen_spec[1], screen_spec[2], screen_spec[0]), (minimap_spec[1], minimap_spec[2], minimap_spec[0]), env.observation_spec()["player"]))
            '''
            
            remote.send(env.observation_spec()[0])

            # remote.send((env.action_spec().functions[data], env.observation_spec()["screen"], env.observation_spec()["minimap"]))
            # check https://github.com/deepmind/pysc2/blob/master/docs/environment.md for detaled imformation
            # non spatial features are available actions
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        env_spec = self.remotes[0].recv()
        # print (test)
        # input()
        # self.screen_space, self.minimap_space, self.ns_space, self.action_space = self.remotes[0].recv()
        self.screen_space, self.minimap_space, self.ns_space, self.action_space = \
            (env_spec["feature_screen"][1], env_spec["feature_screen"][2], env_spec["feature_screen"][0]), (env_spec["feature_minimap"][1], env_spec["feature_minimap"][2], env_spec["feature_minimap"][0]), env_spec["player"], (len(actions.FUNCTIONS),)

    def step(self, actions, arg_dicts):
        for remote, action, arg_dict in zip(self.remotes, actions, arg_dicts):
            remote.send(('step', construct_action(self.screen_space, action, arg_dict)))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return concat_obs(obs, infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return concat_obs(obs, infos)

    def step_async(self):
        return None

    def step_wait(self):
        return None

    def save_replay(self, path, epoch):
        for idx, remote in enumerate(self.remotes):
            remote.send(('save_replay', {"path": path, "epoch": epoch, "env_idx": idx}))
        return None

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def nenvs(self):
        return len(self.remotes)

    @property
    def base_action_count(self):
        return self.action_space[0]


def construct_action(screen_space, act_id, selected_args):
    # Set act_id and act_args
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
        if arg.name in ('screen', 'minimap', 'screen2'):
            act_args.append([int(selected_args[arg.name] % screen_space[1]), int(selected_args[arg.name] // screen_space[1])])
        else:
            act_args.append([selected_args[arg.name]])
    return [actions.FunctionCall(act_id, act_args)]
    '''
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
        if arg.name in ('screen', 'minimap', 'screen2'):
            act_args.append([int(pos % screen_space[1]), int(pos // screen_space[1])])
        else:
            act_args.append([0])
    return [actions.FunctionCall(act_id, act_args)]
    '''

'''
def step(action, arg_dict):
    return self._process_state(self._env.step(self._construct_action(action, arg_dict))[0])
'''

def concat_obs(obs, infos):
    screen, minimap, ns, reward, available_actions, last = [], [], [], [], [], []
    for o, i in zip(obs, infos):
        screen.append(o["screen"])
        minimap.append(o["minimap"])
        ns.append(o["ns"])
        reward.append(i["reward"])
        available_actions.append(i["available_actions"])
        last.append(i["last"])

    # screen, minimap, ns, reward, available_actions, last = np.concatenate(screen, axis=0), np.concatenate(minimap, axis=0), np.concatenate(ns, axis=0), np.concatenate(reward, axis=0), np.concatenate(available_actions, axis=0), np.concatenate(last, axis=0)

    ob_stack = {"screen": screen, "minimap": minimap, "ns": ns}
    info_stack = {"reward": reward, "available_actions": available_actions, "last": last}

    return ob_stack, info_stack

def process_state(state):
    sc2_screen = state.observation["feature_screen"]
    sc2_minimap = state.observation["feature_minimap"]

    processed_screen = np.transpose(sc2_screen, [1, 2, 0])
    processed_minimap = np.transpose(sc2_minimap, [1, 2, 0])

    '''
    processed_screen = []
    processed_screen.append(process_catagorical_map(sc2_screen[_SCREEN_PLAYER_RELATIVE], features.SCREEN_FEATURES[_SCREEN_PLAYER_RELATIVE].scale))
    processed_screen.append(process_catagorical_map(sc2_screen[_SCREEN_SELECTED], features.SCREEN_FEATURES[_SCREEN_SELECTED].scale))
    processed_screen = np.concatenate(processed_screen, axis=2)
    

    processed_minimap = []
    processed_minimap.append(process_catagorical_map(sc2_minimap[_MINIMAP_PLAYER_RELATIVE], features.MINIMAP_FEATURES[_MINIMAP_PLAYER_RELATIVE].scale))
    processed_minimap.append(process_catagorical_map(sc2_minimap[_MINIMAP_SELECTED], features.MINIMAP_FEATURES[_MINIMAP_SELECTED].scale))
    processed_minimap.append(process_catagorical_map(sc2_minimap[_MINIMAP_VISIBILITY], features.MINIMAP_FEATURES[_MINIMAP_VISIBILITY].scale))
    processed_minimap.append(process_catagorical_map(sc2_minimap[_MINIMAP_CAMERA_ID], features.MINIMAP_FEATURES[_MINIMAP_CAMERA_ID].scale))
    processed_minimap = np.concatenate(processed_minimap, axis=2)
    '''

    # forbid_actions = [1, 13, 3, 4, 332, 333, 334, 452]

    ns = state.observation["player"]
    
    available_actions = np.zeros([len(actions.FUNCTIONS)], dtype=np.float32)
    available_actions[state.observation["available_actions"]] = 1
    # available_actions[forbid_actions] = 0
    '''
    ns = np.zeros([len(actions.FUNCTIONS)], dtype=np.float32)
    ns[state.observation["available_actions"]] = 1
    ns[forbid_actions] = 0
    '''
    
    '''
    is_last = False
    if state.last():
        is_last = True
    '''

    ob = {"screen": processed_screen, "minimap": processed_minimap, "ns": ns}
    info = {"reward": state.reward, "available_actions": available_actions, "last": state.last()}

    return ob, info

def process_catagorical_map(input_map, scale):
    '''
    new_size = input_map.shape[0] * input_map.shape[1]
    new_map = np.zeros((new_size, scale))
    input_map = input_map.flatten()
    new_map[np.arange(new_size), input_map] = 1
    return new_map
    '''
    
    # return tf.one_hot(input_map, depth=scale).eval()
    new_map = np.zeros([input_map.shape[0], input_map.shape[1], scale], dtype=np.float32)
    for c in range(scale):
        index_y, index_x = (input_map == c).nonzero()
        new_map[index_y, index_x, c] = 1
    return new_map
    
   
def save_replay(epoch):
    # self._env.save_replay(self._replay_dir)
    # now = datetime.datetime.utcnow().replace(microsecond=0)
    replay_filename = "%s_%d_%d.SC2Replay" % (
    os.path.splitext(os.path.basename(self._map_name))[0],
    epoch,
    self._index)
    if not gfile.Exists(self._replay_dir):
        gfile.MakeDirs(self._replay_dir)
    replay_path = os.path.join(self._replay_dir, replay_filename)
    with gfile.Open(replay_path, "wb") as f:
        f.write(self._env._controllers[0].save_replay())