from math import inf
from random import choice, random, randrange
import gymnasium as gym
from PIL import Image
from typing import Any

env = gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="8x8",
    is_slippery=True,
    success_rate=0.95,
    reward_schedule=(1, 0, 0),
    render_mode="rgb_array"
)

observation, info = env.reset()

MAX_GAMES = 50000
GAME_NUMBER = 0


Q_VALUES: dict[(Any, int), float] = {}

def behaviour(observation, q_values, epsilon):
    action = env.action_space.sample()
    if random() < epsilon:
        return action
    else:
        optimal_Q_AS = 0
        for potential_action in range(env.action_space.n):
            potential_action = int(potential_action)
            if (observation, potential_action) in q_values and q_values[(observation, potential_action)] > optimal_Q_AS:
                optimal_Q_AS = q_values[(observation, potential_action)]
                action = potential_action
        return action

rewards = []
step=0
frames = []
action =  behaviour(observation, Q_VALUES, epsilon=0.3)
STEP_SIZE = 0.1
GAMMA = 0.95

epsilon = 1
delta_epsilon = epsilon / MAX_GAMES
if GAME_NUMBER % 1000 == 0:
    frame = env.render()
    frames.append(Image.fromarray(frame))
while GAME_NUMBER < MAX_GAMES:
    step+=1

    old_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)
    if GAME_NUMBER % 1000 == 0:
        frame = env.render()
        frames.append(Image.fromarray(frame))
    
    Q_oldAS = 0
    if (old_observation, action) in Q_VALUES:
        Q_oldAS = Q_VALUES[(old_observation, action)]

    optimal_Q_AS = 0
    if terminated or truncated:
        Q_VALUES[(old_observation, int(action))] = Q_oldAS + STEP_SIZE * (reward - Q_oldAS)
    else:
        for potential_action in range(env.action_space.n):
            potential_action = int(potential_action)
            if (observation, potential_action) in Q_VALUES and Q_VALUES[(observation, potential_action)] > optimal_Q_AS:
                optimal_Q_AS = Q_VALUES[(observation, potential_action)]
        Q_VALUES[(old_observation, int(action))] = Q_oldAS + STEP_SIZE * (reward + (GAMMA*optimal_Q_AS)-Q_oldAS)
        action =  behaviour(observation, Q_VALUES, epsilon=epsilon)

    rewards.append(reward)
    if terminated or truncated:
        epsilon -= delta_epsilon
        print(sum(rewards))
        if GAME_NUMBER % 1000 == 0:
            frames[0].save(
                f"exports/frozen_lake/games/game_{GAME_NUMBER}.gif",
                save_all=True,
                append_images=frames[1:],
                duration=500,  # Duration of each frame in milliseconds
                loop=0
            )
        GAME_NUMBER += 1
        observation, info = env.reset()
        if GAME_NUMBER % 1000 == 0:
            frame = env.render()
            frames.append(Image.fromarray(frame))
        step = 0
        frames = []
        rewards = []
        action =  behaviour(observation, Q_VALUES, epsilon=0.3)


print(Q_VALUES)