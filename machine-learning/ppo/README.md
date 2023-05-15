## Proximal Policy Optimization (PPO)

Following [Reinforcement Learning Course: Intro to Advanced Actor Critic Methods - YouTube](https://www.youtube.com/watch?v=K2qjAixgLqk)

### 1. Introduction

In Actor-Critic, sometimes the performance could fall off a cliff. The agent would be doing really well for a while and an update to the neural network would cause the agent to lose its understanding of playing the game. This happens because Actor-Critic methods are sensitive to perturbations. The reason being that a small change in the policy can lead to a large jump in policy space.

PPO addresses this by limiting update to policy network. We base the update each step on on the ratio of the new policy to the old policy. We constraint that ratio to a specific range to make sure we are not making huge steps and parameter space for our neural network.

Also we have to take into account the advantage of how valuable each state is, and the reason being that we want to encourage the agent to take actions that lead to higher rewards over time.

Taking into account the advantage can cause the loss function to grow a little bit too large so we are clipping the loss function and taking the lower bound with the minimum function.

Instead of keeping track of million trasitions and then sampling a subset of those at random, we keep track of a fixed length trajectory of memories.

We use multiple network updates per data sample and use mini-batch SGD

#### Implementation notes

Memory indices = [0, 1, 2, ..., 19]

Let's say we have a mini-batch size of 5. Batches start at multiples of batch_size at positions [0, 5, 10, 15]

We shuffle memories then take batch size chunks.

Two distinct networks instead of shared inputs.

Critic evaluates states (not s,a pairs)

Actor decides what to do based on current state.
- Network outputs probs (softmax) for a categorical distribution
- Exploration die to nature of distribution

Memory is fixed to length T (say, 20) steps. Note that T is much smaller than the length of an episode.

Track states, actions, rewards, values, and dones, log probs of selecting actions.

Perform 4 epochs of updates per mini-batch.

Update rule for actor is complex:

$L^{CPI}(\theta) = \hat{E}_t[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}A_t]$ = $\hat{E}_t[r_t(\theta)\hat{A}_t]$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ and $\hat{A}_t = \frac{A_t}{\sigma(A_t)}$

- CPI stands for Conservative Policy Iteration.
- $\hat{E}_t$ refers to the empirical or sample expectation at timestep $t$.
- $\hat{A}_t$ represents the estimated Advantage at time step $t$, in practice, we often don't know the true advantage function, so we use an estimate, denoted with a hat.
- $\pi$ is usually used to denote the policy in reinforcement learning, which is a mapping from states to actions. In other words, it's the strategy that the agent uses to determine what action to take in a given state.
- $\theta$ represents the parameters of the policy. In the case of a neural network policy, these would be the weights and biases of the network.
- $(a_t|s_t)$ is the conditional probability of taking action $a$ at time $t$ given state $s$ at time $t$.
- $\pi_\theta(a_t|s_t)$ is the probability of taking action $a_t$ in state $s_t$ under the current policy $\pi$ parameterized by $\theta$.

Define epsilon (say, ~0.2) for clip/min operations:
$L^{CLIP}(\theta) = \hat{E}_t[min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$
where:
- $L^{CLIP}(\theta)$: This is the objective function that the Proximal Policy Optimization (PPO) algorithm tries to maximize. $\theta$ represents the parameters of the policy. Maximizing this function makes the policy better over time.
- $\hat{E}_t[...]$: This represents the expected value of the term inside the brackets, approximated over a batch of data at time step $t$. The expectation is taken with respect to the probability distribution of trajectories (sequences of states, actions, and rewards) given by the current policy.
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: This is the probability ratio of the new policy $\pi_\theta$ to the old policy $\pi_{\theta_{old}}$. If the new policy is more likely to take an action than the old policy was, this ratio is greater than 1. If it's less likely, the ratio is less than 1.
- $\hat{A}_t$: This is the estimated advantage at time step $t$. The advantage function measures how much better or worse an action is compared to the average action in that state. A positive advantage indicates that the action is better than average, while a negative advantage suggests it's worse than average.
- $min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)$: This is the key part of PPO. The objective function is defined as the minimum of two terms. The first term is just the probability ratio times the advantage, as in the standard policy gradient objective. The second term is the same, but the probability ratio is clipped to be within $[1-\epsilon, 1+\epsilon]$. The use of the minimum means that the update is "pessimistic", favoring lower values. This helps to prevent the new policy from straying too far from the old policy.

Which leads us to the advantage at each time step:
$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + ... + (\gamma\lambda)^{T-t+1}\delta_{T-1}$
where:
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$: This is the TD error at time step $t$. It's the difference between the reward $r_t$ received at time step $t$ and the estimated value of the current state $V(s_t)$, plus the discounted estimated value of the next state $\gamma V(s_{t+1})$. The advantage is the sum of the Temporal Difference (or TD) errors for all future time steps, weighted by a discount factor $\gamma$ and another parameter $\lambda$ called the GAE parameter.
- $\lambda$: This is the GAE parameter. It's a number between 0 and 1 that controls the extent to which the agent looks into the future. A value of 0 means that only the current TD error is used (i.e. no bootstrapping), while a value of 1 means that the agent looks as far as possible into the future (i.e. infinite bootstrapping). In practice, a value of 0.95 is often used.

Critic loss is more straightforward:

Return = advantage + critic value (from mem)

$L_{critic}$ = MSE(return - critic value (from network))

Total loss is sum of clipped actor and critic:
$L^{CLIP+VF+S}_t(\theta) = \hat{E}_t[L^{CLIP}_t(\theta) - c_1L^{VF}_t(\theta) + c_2S[\pi_\theta](s_t)]$
where:
- $L^{CLIP+VF+S}_t(\theta)$: This is the combined objective function that the Proximal Policy Optimization (PPO) algorithm tries to maximize. $\theta$ represents the parameters of the policy.
- $\hat{E}_t[...]$: This again represents the expected value, similar to the previous formula. The expectation is taken with respect to the probability distribution of trajectories (sequences of states, actions, and rewards) given by the current policy.
- $L^{CLIP}_t(\theta)$: This is the same PPO clipping objective function as before.
- $c_1L^{VF}_t(\theta)$: $c_1$ is a constant that weighs the contribution of the value function loss, $L^{VF}_t(\theta)$. The value function is a prediction of future rewards from a given state, and this term aims to minimize the error between the predicted and actual future rewards. This encourages the policy to favor actions leading to higher future rewards.
- $c_2S\pi_\theta$: $c_2$ is another constant that weighs the contribution of the entropy bonus, $S\pi_\theta$. Entropy is a measure of randomness or unpredictability. Adding the entropy of the policy to the objective function encourages exploration by discouraging the policy from becoming too deterministic (i.e., always selecting the same action in a given state).
- $S$ only comes into play when we have a deep neural network with shared lower layers and actor-critic outputs at the top. (Not implemented in this tutorial because we are doing two separate networks actor-critic)
- $C_1$ = 0.5

Note that we are doing gradient ascent.

#### Data structures

- class for replaybuffer: lists (or numpy arrays)
- class for actor network, class for critic network
- class for agent
- main loop to train and evaluate