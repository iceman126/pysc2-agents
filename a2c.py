import gym
import numpy as np
import tensorflow as tf
import subprocess
from agent import Agent

def learn(network, log_path, env_function, num_cpu=4, nsteps=10, total_timesteps=int(80e6), optimizer="rmsprop", vl_coef=1.0, ent_coef=1e-3, max_grad_norm=0.5, ar=True, lr=7e-4, gamma=0.99):
    sess = tf.InteractiveSession()
    agent = Agent(sess, log_path, env_function, num_cpu, network, ar, lr, optimizer, ent_coef, vl_coef, max_grad_norm, nsteps=nsteps, gamma=gamma)
    training_steps = 5000
    epochs = (total_timesteps // training_steps) + 1
    evaluating_episodes = 20
    sess.run(tf.global_variables_initializer())
    for epoch in range(0, epochs):
        # Training
        agent.training_process(epoch, training_steps, 0.01)  # Note: the epsilon value is just a dump value, we're not using it
        agent.save_model(epoch)
        # Evaluation
        eval_rewards = agent.evaluating_process(epoch, evaluating_episodes)
        
        log_file = open("%seval.log" % log_path, "a+")
        log_file.write("%d\t%f\t%f\t%f\n" % (training_steps * (epoch + 1), np.max(eval_rewards), np.mean(eval_rewards), np.min(eval_rewards)))
        log_file.close()
