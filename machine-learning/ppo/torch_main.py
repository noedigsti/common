"""
pip install pygame moviepy gymnasium
"""
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.human_rendering import HumanRendering
import numpy as np
from torch_ppo import Agent
from utils import plot_learning_curve
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    """
    CartPole-v1 has the following conditions but somehow the agent is able to play the game even after violating these conditions. Idk how.
    - The pole is more than 15 degrees from vertical.
    - The cart moves more than 2.4 units from the center.
    - The episode length is greater than 500.
    
    I have capped the score to 10000 since the agent is able to play the game easily after a certain point.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    if args.view:
        wrapped = HumanRendering(env)
    if args.record:
        """
        RecordVideo wrapper records the video of the environment. For some reason, it does not work with HumanRendering wrapper. Which means that you cannot record the video of the environment while viewing the agent play the game.

        Also the video length is capped at 10 seconds and I don't know how to increase it. I have tried changing the video_length parameter but it doesn't work.
        """
        args.view = False
        env = RecordVideo(
            env, "video", episode_trigger=lambda x: x % 1 == 0, video_length=10000
        )

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
    if args.load_model or args.record:
        agent.load_models()
    if args.record:
        n_game = 3
    else:
        n_game = 300
    figure_file = "plots/cartpole.png"
    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    for i in range(n_game):
        if args.view:
            observation, _ = wrapped.reset()
        else:
            observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            if args.view:
                observation_, reward, done, _, info = wrapped.step(action)
            else:
                observation_, reward, done, _, info = env.step(action)
            n_steps += 1
            score += reward
            if not args.record:
                agent.remember(observation, action, prob, val, reward, done)
                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
            observation = observation_
            if score >= 10000:
                done = True
        if not args.record:
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
            print(
                f"episode {i}, score {score:.2f}, avg_score {avg_score:.2f}, time_steps {n_steps}, learning_steps {learn_iters}"
            )
        else:
            print(f"Episode {i}, Score: {score}")
        if args.record:
            env.close()

    if not args.record:
        if args.view:
            wrapped.close()
        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)
    env.close()
