import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOMemory:
    def __init__(self, batch_size: int):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        self.batch_size = batch_size

    def store_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        """
        This method generates batches of states, actions, log probabilities, values,
        rewards, and done flags to be used for training the agent.

        Returns:
        --------
            np.array: states in a numpy array
            np.array: actions in a numpy array
            np.array: log probabilities in a numpy array
            np.array: values in a numpy array
            np.array: rewards in a numpy array
            np.array: done flags in a numpy array
            list: batches of indices indicating the start of each batch
        """

        # Get the total number of states
        n_states = len(self.states)

        # Create an array with the starting index of each batch
        batch_start = np.arange(0, n_states, self.batch_size)

        # Create an array of indices from 0 to n_states
        indices = np.arange(n_states, dtype=np.int64)

        # Shuffle the indices to create randomness
        np.random.shuffle(indices)

        # Create batches by selecting indices for each batch
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        # Return all states, actions, log probabilities, values, rewards,
        # done flags, and the batches of indices as numpy arrays
        return (
            np.array(self.states),  # Convert list of states to numpy array
            np.array(self.actions),  # Convert list of actions to numpy array
            np.array(
                self.log_probs
            ),  # Convert list of log probabilities to numpy array
            np.array(self.values),  # Convert list of values to numpy array
            np.array(self.rewards),  # Convert list of rewards to numpy array
            np.array(self.dones),  # Convert list of done flags to numpy array
            batches,  # List of index batches
        )

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []


class ActorNetwork(nn.Module):
    def __init__(
        self,
        n_actions: T.Tensor,
        input_dims,
        alpha,
        fc1_dims=256,
        fc2_dims=256,
        checkpoint_dir="tmp/ppo",
    ):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, "actor_torch_ppo")
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)  # Get action probabilities
        dist = Categorical(dist)  # Create a categorical distribution
        return dist  # Return the distribution

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims: T.Tensor,
        alpha: float,
        fc1_dims=256,
        fc2_dims=256,
        checkpoint_dir="tmp/ppo",
    ):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, "critic_torch_ppo")

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor):
        value = self.critic(state)
        return value  # (batch_size, 1)

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(
        self,
        n_actions,
        input_dims,
        gamma=0.99,
        alpha=3e-4,
        policy_clip=0.2,
        batch_size=64,
        N=2048,
        n_epochs=10,
        gae_lambda=0.95,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
        self.batch_size = batch_size
        self.N = N
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    def remember(self, state, action, probs, values, reward, done):
        self.memory.store_memory(state, action, probs, values, reward, done)

    def choose_action(self, observation):
        observation = np.array(observation)
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist: Categorical = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        return action, probs, value

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def learn(self):
        """
        Trains the actor and critic networks.

        This method implements the core of the PPO algorithm. It loops over the number
        of specified epochs, generates batches of experiences from the memory, calculates
        the advantage for each experience, and uses these to update the actor and critic networks.

        It first updates the actor network by calculating the policy loss with both
        clipped and unclipped probability ratios, taking the minimum of these as the final
        policy loss. It then calculates the value loss for the critic network as the mean
        squared error between the estimated and actual returns.

        The total loss is a combination of the policy (actor) loss and value (critic) loss.
        This total loss is used to perform a backward pass and an optimization step for
        both the actor and critic networks.

        Finally, the memory is cleared, ready for the next batch of experiences.

        Note: This method assumes that the number of epochs, the discount factor, lambda
        for GAE, policy clip range, and device (cpu or gpu) are available as attributes
        of the self object.
        """
        # Loop over training epochs
        for _ in range(self.n_epochs):
            # Generate batches of experiences from memory
            (
                state_arr,
                action_arr,
                old_log_probs_arr,
                values_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            # Get the values from the memory
            values = values_arr
            # Initialize an array to store advantage values
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculate the advantages
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                # Compute the Generalized Advantage Estimation (GAE)
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            # Convert advantage to PyTorch tensor and move to the device
            advantage = T.tensor(advantage).to(self.actor.device)
            # Convert values to PyTorch tensor and move to the device
            values = T.tensor(values).to(self.actor.device)

            # Train on each batch
            for batch in batches:
                # Convert states, old log probabilities, and actions to PyTorch tensors and move to the device
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_log_probs = T.tensor(old_log_probs_arr[batch], dtype=T.float).to(
                    self.actor.device
                )
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                # Get policy distribution and value estimate for current states
                dist = self.actor(states)
                critic_value = self.critic(states)

                # Ensure critic_value is the right shape
                critic_value = T.squeeze(critic_value)

                # Compute new log probabilities under the current policy
                new_probs = dist.log_prob(actions)
                # Compute the ratio of the new and old probabilities
                prob_ratio = new_probs.exp() / old_log_probs.exp()

                # Compute the unclipped and clipped policy losses
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )

                # Compute the actor loss by taking the minimum of the unclipped and clipped losses
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Compute target values for the critic
                returns = advantage[batch] + values[batch]
                # Compute the critic loss as the mean squared error between the estimated and target values
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # Combine the actor and critic losses
                total_loss = actor_loss + 0.5 * critic_loss
                # Zero the gradients before the backward pass
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                # Perform the backward pass to compute gradients
                total_loss.backward()
                # Perform the update steps for both the actor and critic networks
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            # Clear the memory after updating
            self.memory.clear_memory()
