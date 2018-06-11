#!/usr/bin/env python
import os
import time

from a2c import learn
from pysc2 import maps
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
from absl import flags
from misc import set_global_seeds, make_path
from common.vec_env.subproc_vec_env import SubprocVecEnv

FLAGS = flags.FLAGS
flags.DEFINE_integer("screen_size", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_size", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("num_timesteps", int(10e7), "Total agent steps.")
flags.DEFINE_integer("num_cpu", 8, "Number of cpus.")
flags.DEFINE_integer("batch_steps", 16, "Agent steps of a agent per batch")
flags.DEFINE_float("lr", 1e-4, "Learning rate")
flags.DEFINE_float("ent_coef", 1e-3, "Entropy coefficient")
flags.DEFINE_float("vl_coef", 1e-3, "Value loss coefficient")
flags.DEFINE_float("max_grad_norm", 0.5, "Max Gradient Norm")
flags.DEFINE_string("network", "fullyconv", "Learning rate schedule")
flags.DEFINE_boolean("ar", False, "Whether using auto-regressive network")
flags.DEFINE_string("lrschedule", "constant", "Learning rate schedule")
flags.DEFINE_string("optimizer", "rmsprop", "Optimizer for training")
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.mark_flag_as_required("map")

def train(map_name, num_timesteps, batch_steps, seed, network, ar, lr, lrschedule, screen_size, minimap_size, step_mul, num_cpu, optimizer, ent_coef, vl_coef, max_grad_norm):
    maps.get(map_name)  # Assert the map exists.

    log_path = './experiments/%s/' % (time.strftime("%m%d_%H%M_") + map_name)
    make_path(log_path)
    make_path("%sreplay" % log_path)

    def make_env(rank):
        def _thunk():
            agent_interface = features.parse_agent_interface_format(
                feature_screen=64,
                feature_minimap=64
            )
            env = sc2_env.SC2Env(map_name=map_name,
                                 step_mul=step_mul,
                                 agent_interface_format=agent_interface,
                                 # screen_size_px=(screen_size, screen_size),
                                 # minimap_size_px=(minimap_size, minimap_size),
                                 visualize=False)
            return env
        return _thunk

    set_global_seeds(seed)

    log_file = open("%sconfig.log" % log_path, "a+")
    log_file.write("Map Name: %s\n" % map_name)
    log_file.write("Optimizer: %s\n" % optimizer)
    log_file.write("Network: %s\n" % network)
    log_file.write("Learning Rate: %f\n" % lr)
    log_file.write("Entropy Coefficient: %f\n" % ent_coef)
    log_file.write("Value Function Coefficient: %f\n" % vl_coef)
    log_file.write("Maximum Gradient Norm: %f\n" % max_grad_norm)
    log_file.write("Screen Size: %d\n" % screen_size)
    log_file.write("Minimap Size: %d\n" % minimap_size)
    log_file.write("Batch Steps: %d\n" % batch_steps)
    log_file.close()

    learn(network, log_path, make_env, total_timesteps=num_timesteps, nsteps=batch_steps, ent_coef=ent_coef, max_grad_norm=max_grad_norm, optimizer=optimizer, vl_coef=vl_coef, ar=ar, lr=lr, num_cpu=num_cpu)

def main(unused_argv):
    train(FLAGS.map, FLAGS.num_timesteps, FLAGS.batch_steps, int(time.time()), FLAGS.network, FLAGS.ar, FLAGS.lr,
        FLAGS.lrschedule, FLAGS.screen_size, FLAGS.minimap_size, FLAGS.step_mul,
        FLAGS.num_cpu, FLAGS.optimizer, FLAGS.ent_coef, FLAGS.max_grad_norm, FLAGS.vl_coef)

if __name__ == "__main__":
    app.run(main)

'''
if __name__ == '__main__':
    main()
'''
