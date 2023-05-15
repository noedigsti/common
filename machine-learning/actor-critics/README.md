#### Action

Action is the output of the actor network. It is a probability distribution over the actions. For example, if the action is a discrete action space, then the action is a probability distribution over the discrete actions. If the action is a continuous action space, then the action is a probability distribution over the continuous actions.

For example:
LunarLander-v2 has a discrete action space of 4 actions that are do nothing, fire left engine, fire main engine, fire right engine. The action is a probability distribution over these 4 actions.

discrete action space. softmax activation. categorical distribution.

#### Rewards

The reward is the reward that the agent gets from the environment. The reward is a scalar value and is used to calculate the loss of the actor and critic networks.

#### Dynamics

The dynamics is the transition function of the environment. It is the function that takes in the current state and action and returns the next state and reward.

$P(s', r | s, a)$ -> Probability of getting to state s' and reward r given state s and action a.

#### Expected return

Takes into account all possible outcomes and multiplies them by their probability of occurring. It is the sum of all the rewards that the agent gets from the environment.

#### Episode

An episode is a single sequence of interactions between an agent and the environment, representing a complete trajectory from the initial state to a terminal state.

$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... = \sum_{t=0}^{T-1} R_{t+1}$

What about continuous tasks? The expected return is infinite. So we use the discounted return.

#### Discounted return

We reduce the contribution of rewards that are further in the future. We do this by multiplying the reward by a discount factor $\gamma$ (hyperparameter, 0 <= $\gamma$ <= 1)

$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

Also reflects the fact that rewards received in the future are worth less than rewards received in the present.

$G_t = R_{t+1} + \gamma G_{t+1}$

Consistent with Markov Decision Process (MDP) formulation.

#### Value function

The value function is the expected return starting from a particular state. It is the expected return if the agent starts in that state and then follows the policy for all future actions.

$v_{\pi}(s) = E_{\pi}[G_t | S_t = s] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s]$

These are estimated with neural networks by sampling returns from the environment, because neural networks are universal function approximators.

## Actor-Critic

Actor-Critic is a policy gradient method. It is a combination of policy-based and value-based methods. It has two deep neural networks: an actor network and a critic network. The actor network is the policy network and the critic network is the value network.

The actor selects an action given a state. The critic evaluates the value of the state. The resulting action is then evaluated by the environment and the critic. The critic gives feedback to the actor on how good the action was.

From a practical perspective, we update the weights of our neural network at each time step because Actor-Critic belongs to a class of algorithms called Temporal Difference (TD) Learning. TD methods update the weights of the neural network at each time step. Each network (Actor and Critic) has its own loss function.

$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

<!-- Fact check -->
Critic Loss: $L = \frac{1}{2} \delta_t^2$
<!-- Fact check -->
Actor Loss: $L = -log(\pi(A_t|S_t)) \delta_t$

## Actor-Critic Algorithm

<!-- This is actually PPO-Clip -->
1. the actor network (policy network) and the critic network (value network) with random weights.
2. Repeat for a number of episodes:
   1. Reset the environment to get the initial state.
   2. Collect data for a number of steps:
   3. For each step until the episode is done:
      1. Select an action from the actor network (using the current policy).
      2. Perform the action and observe the next state and reward.
      3. Store the transition (state, action, reward, next state) in a memory buffer.
      4. Update the current state to be the next state.
   4. Once we have collected enough data, calculate the returns (cumulative discounted rewards) for each stored transition.
   5. For a certain number of optimization epochs, do the following:
      1. Calculate the values for each state in our collected data using the critic network.
      2. Calculate the advantages for each state-action pair in our collected data (this could be done using the TD error).
      3. Update the weights of the actor network by maximizing the PPO-Clip objective function, which encourages the new policy to stay close to the old policy while still improving according to the advantage function.
      4. Update the weights of the critic network by minimizing the value loss function (for example, the mean squared error between the calculated values and the returns).
   6. Optionally, update the old policy to be the current policy.

<!-- Fact check the 2nd half -->
With Actor-Critic, you may see some fluctuations in the rewards, but the overall trend should be an increase in the rewards. Another thing you may see is the score increasing and then falling off a cliff because Actor-Critic is a high variance algorithm. This means that it can have a lot of variance in the rewards that it gets. This is because the policy is constantly changing and the critic is constantly changing. This can lead to the policy diverging and the agent getting worse and worse rewards. To fix this, we can use a technique called entropy regularization.

#### Implementation

- Use one network, common lower layers, two heads/outputs (actor and critic)

## Deep Deterministic Policy Gradient (DDPG)

DDPG is an off-policy actor-critic algorithm that uses a deterministic policy. It is an off-policy algorithm because it uses two different policies: a policy for exploration and a policy for exploitation. It is an actor-critic algorithm because it uses two deep neural networks: an actor network and a critic network.

The actor network is the policy network and the critic network is the value network. The actor network outputs a deterministic action given a state. The critic network outputs the value of the state-action pair.

Actor decides what to do on current state.
- Network outputs action values, not probabilities.
- Noise for exploration.

#### Implementation notes

Update rule for actor network:
- Randomly sample a batch of transitions from the replay buffer.
- Use actor to calculate the action values for each state in the batch.
- Plug those actions into critic to calculate the value for each state-action pair in the batch.
- Take the gradient w.r.t. the actor network weights.

Update rule for critic network:
- Randomly sample a batch of transitions from the replay buffer. (states, new states, actions, rewards)
- Use target actor to calculate the next actions for each new state in the batch.
- Plug those actions into target critic to calculate the next values for each new state-action pair in the batch.
- Plug states, actions into critic and take diff with target.

Target networks:
- Randomly initialize critc network Q(s,a|$\theta^Q$) and actor $\mu$(s|$\theta^\mu$) with weights $\theta^Q$ and $\theta^\mu$.
- Initialize target network $Q'$ and $\mu'$ with weights $\theta^{Q'} \leftarrow \theta^Q$, $\theta^{\mu'} \leftarrow \theta^\mu$.
- Update the target networks:
- Weights of the target critic network: $\theta^{Q'} \leftarrow \tau \theta^Q + (1 - \tau) \theta^{Q'}$
- Weights of the target actor network: $\theta^{\mu'} \leftarrow \tau \theta^\mu + (1 - \tau) \theta^{\mu'}$

Data structures we will need:
- class for replay buffer: numpy array
- class for actor network, class for critic network
- class for agent (ties everything together): actor network, critic network, replay buffer, noise process
- main loop to train and evaluate agent

The point of the buffer is to store transitions. We will randomly sample transitions from the buffer to train the agent. The buffer will store the following information for each transition: state, action, reward, next state, done. 

Since the action space is continuous, n_actions is just the dimension of the action space.

target_actor, target_critic will do soft updates and not gradient descent.