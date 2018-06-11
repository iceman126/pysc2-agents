# pysc2-agent
This project is a re-implementation of paper "StarCraft II: A New Challenge for Reinforcement Learning". The learning algorithm is A2C and both Atari-net and FullyConv are implemented. Furthermore, The atari-net can be trained in auto-regressive manner.  

## Requirments
- python 3 or above
- [pysc2](https://github.com/deepmind/pysc2) 2.0.1
- [tensorflow](https://www.tensorflow.org/) or tensorflow-gpu

## TODOs
- Load trained models
- Build FullyConv in auto-regressive manner
- Save videos of evaluation epsiodes (New feature of pysc2 2.0)

## Run an agent
Run a agent in MoveToBeacon mini-game with 1e-4 learning rate.

<code>python3 run_sc2.py --map MoveToBeacon --ent_coef 1e-3 --lr 1e-4 --num_timesteps 600000 --num_cpu 8 --vl_coef 1.0 --max_grad_norm 0.5 --network atari --ar</code>
### Arguments
-  <code>--map</code>: The map you want to train on.
-  <code>--ent_coef</code>: Entropy coeffient.
-  <code>--vl_coef</code>: Weight of value loss.
-  <code>--lr</code>: Learning rate.
-  <code>--optimizer</code>: Optimizer for updating the weights. Available options: <code>rmsprop</code> and <code>adam</code>
-  <code>--num_timesteps</code>: Total training steps.
-  <code>--num_cpu</code>: Number of enviornments to run simultaneously.
-  <code>--max_grad_norm</code>: Max gradient norm, for gradient clipping.
-  <code>--network</code>: Type of network. The available options are: <code>atari</code> and <code>fullyconv</code>
-  <code>--ar</code>: This is a boolean. Decide whether to build the network in auto-regressive manner. (This is only available when using Atari-net)

##  Performance
**Best mean scores:**

|  |Ours|DeepMind|
|--|--|--|
|MoveToBeacon|25|26|
|DefeatRoaches|91|101|
|CollectMineralShards||104|