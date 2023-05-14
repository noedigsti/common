import gymnasium as gym
from gymnasium.wrappers.human_rendering import HumanRendering
import numpy as np
from torch_ppo import Agent
from utils import plot_learning_curve

if __name__ == "__main__":
    # CartPole-v1 has the following conditions:
    # The pole is more than 15 degrees from vertical.
    # The cart moves more than 2.4 units from the center.
    # The episode length is greater than 500.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    wrapped = HumanRendering(env)
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )
    n_game = 300
    figure_file = "plots/cartpole.png"
    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    for i in range(n_game):
        observation, _ = wrapped.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print(
            f"episode {i}, score {score:.2f}, avg_score {avg_score:.2f}, time_steps {n_steps}, learning_steps {learn_iters}"
        )

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
