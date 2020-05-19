# Author: Tanmay Aggarwal

# Project Continuous Control

### Overview

In this project, I use the off-policy Deep Deterministic Policy Gradient (DDPG) algorithm to train a double-jointed arm to move to a target location simulated within a Unity3D environment. There are three main files in this project:
1. Continuous_Control.ipynb
2. ddpg_agent.py
3. model.py

The Jupyter Notebook 'Continuous_Control.ipynb' is the main file used to train and run the agent. The following macro parameters are used in the training process:
- n_episodes: 2000
- max_t: 1000
- reset_time: 1000
- print_every: 100
- random_seed: 2

The 'ddpg_agent.py' file is used to define the Agent class, the Noise process class, as well as the ReplayBuffer class. The following hyperparameters are used in the training process:

- BUFFER_SIZE = 1e6                 # replay buffer size
- BATCH_SIZE = 128                  # minibatch size
- GAMMA = 0.98                       # discount factor
- TAU = 1e-3                        # for soft update of target parameters
- LR_ACTOR = 4e-4                   # learning rate of the actor
- LR_CRITIC = 4e-4                  # learning rate of the critic
- WEIGHT_DECAY = 0                  # L2 weight decay
- UPDATE_EVERY = 20               # Frequency of learning
- UPDATE_TIMES = 10               # how many times to update the network each time
- EPSILON = 1.0                   # epsilon for the noise process
- EPSILON_DECAY = 0.999           # decay for epsilon above

For the OUNoise process, the following parameters were used:
- mu = 0.
- Theta = 0.15
- Sigma = 0.05

A replay buffer is used to sample experiences randomly to update the neural network parameters.

The 'model.py' file is where the Actor and the Critic neural network models are defined. These neural networks are being used as function approximators that guide the agent's actions at each time step (i.e., the policy). The Actor model directly maps each state with actions. Meanwhile, the Critic network determines the Q-value for each (state, action) pair. The next-state Q values are calculated with the target value (critic) network and the target policy (actor) network. Meanwhile, the original Q value is calculated with the local value (critic) network, not the target value (critic) network.

The overall architecture of the model is relatively simple with two fully connected hidden layers with 128 and 256 nodes each, respectively. ReLU activation function is used in between each layer in the network. An Adam optimizer is used to minimize the loss function. I have used the Mean Squared Error loss function in this model.

The target networks are updated via "soft updates" controlled by the hyperparameter Tau.

Finally, I have used the Ornstein-Uhlenbeck Process to add noise to the action output in order to encourage exploration in the continuous action space.

## Discussion of the hyperparameters

Training this model requires a delicate balance between exploration (controlled via the OUNoise process) and exploitation (controlled by the learning rates). The final hyperparameters were achieved via an iterative process wherein different values were tested while observing the corresponding scores. Discussion for some of the key hyperparameters is as follows:
- Batch size: the goal here was to find a balance between a batch size which was large enough to meaningfully estimate a gradient while yet being small enough to not overburden the training process.
- Gamma: various values between 0.9-0.99 were tested. Eventually, 0.98 gave the best result ensuring long term rewards were being discounted but within reason.
- Tau: given Tau controls the frequency of soft updates between the local and target networks, a small value was chosen to ensure a slow learning process for the target networks (and avoid significantly large learning steps at any given sample set).
- Learning rates (actor and critic): various values were explored ranging from 1e-3 to 4e-4. Eventually, the model trained most effectively with the final learning rate of 4e-4. No significant improvement was noted when using a slightly different learning rate between the actor and critic.
- UPDATE_EVERY & UPDATE_TIMES: similar to the original paper on DDPG, it was found that the model learns better when the update frequency is limited while each update involved multiple learnings.
- EPSILON & EPSILON_DECAY: The model was found to perform significantly better when the OUNoise process was annealed using an epsilon decay rate.
- OUNoise variables (particularly, sigma): The model performance significantly improved when sigma was reduced so as to avoid a large standard deviation in the OUNoise process (that guided the exploration of the agent). Relatedly, a normal distribution was found to be more effective in the OUNoise process compared to a uniform distribution.


### Plot of rewards

The environment is solved (i.e., the average score of 30 is reached) in 387 episodes.

![Rewards Image](/Plot_of_Rewards.png)

### Future improvements

The following future improvements to this agent should be considered to further improve effectiveness of the agent (i.e., get a higher average score and / or lower training time):
1. Using batch normalization and gradient clipping in the actor neural network to further stabilize learning.
2. Using a deeper neural network and / or a larger batch size to improve the learning process.
3. Implementing other deep reinforcement learning algorithms such as Truncated Natural Policy Gradient (TNPG) or Trust Region Policy Optimization (TRPO). Amongst these, my hypothesis is that TRPO might be advantaged given it allows for more precise control on the expected policy improvement relative to TNPG.
4. Implementing a distributed learning algorithm such as Proximal Policy Optimization (PPO), Asynchronous Advantage Actor-Critic (A3C), or Distributed Distributional Deterministic Policy Gradients (D4PG) to train another version of this problem wherein 20 identical double jointed arms are trained in parallel.
