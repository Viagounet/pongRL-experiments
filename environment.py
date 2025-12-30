import ale_py
import shimmy
import numpy as np
import gymnasium as gym
import imageio

from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from state_utils import frames_to_bw_resized
from image_utils import save_bw, save_rgb
from nn_utils import initialize_linear_nn_params, infer, return_partial_derivatives, update_params_with_grads, mse

env = gym.make("ALE/Pong-v5")

# 2. Réinitialisation de l'environnement
observation, info = env.reset()
frames = []

accumulate_training_examples = []

def generate_value_func_training_examples(rewards: list[float], states: list[np.array], discount_factor: float):
    values = []
    states_for_nn = []
    steps_since_last_reward = 0
    for reward, state in zip(rewards, states):
        states_for_nn.append(state.reshape(-1))
        steps_since_last_reward += 1
        if reward != 0:
            values += [reward * (discount_factor**i) for i in range(steps_since_last_reward)][::-1]
            steps_since_last_reward = 0
    if steps_since_last_reward > 0:
            values += [0.0] * steps_since_last_reward
    return np.array(states_for_nn) / 255, np.array(values).reshape(len(rewards), 1)

states = []
rewards = []

value_function_params = initialize_linear_nn_params((168*128,50,50,50,1))

HYPERPARAMS = {"initial_lr": 1e-3, "final_lr": 0, "lr_regime": "linear"}

def train(x_train, y_train, params, hyperparams, max_steps):
    training_loss_history = []
    for step in tqdm(range(max_steps)):
        pred, activations = infer(x_train, params)
        train_loss = mse(y_train, pred)
        training_loss_history.append(train_loss)
        grads = return_partial_derivatives(params, activations, x_train, y_train, pred)
        params = update_params_with_grads(params, grads, step, max_steps, hyperparams)
    print(train_loss)
    return params

def simulate_from_state_best_policy(env, previous_frames, value_function_params):
    original_state_snapshot = env.unwrapped.clone_state()
    ORIGINAL_PREVIOUS_FRAMES = deepcopy(previous_frames)
    values = []
    for action_type in range(env.action_space.n):
        observation, reward, terminated, truncated, info = env.step(action_type)
        previous_frames.append(observation)
        state = frames_to_bw_resized(previous_frames, 4, (168,128)) 
        value, activations = infer(state.reshape(-1).reshape(1,-1) / 255, value_function_params)
        values.append(value.item())
        env.unwrapped.restore_state(original_state_snapshot)
        previous_frames = ORIGINAL_PREVIOUS_FRAMES

    return values

N_GAMES = 1
action_choice_strategy = "random"
print(f"STARTING GAME NUMBER 1 [{action_choice_strategy}]")
while N_GAMES < 100:
    # if N_GAMES < 4:
    #     action = env.action_space.sample()
    # else:
    # action = env.action_space.sample()
    action_choice_strategy = "best value"
    predicted_values = simulate_from_state_best_policy(env, frames, value_function_params)
    action = predicted_values.index(max(predicted_values))
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(observation)
    state = frames_to_bw_resized(frames, 4, (168,128)) 
    value, activations = infer(state.reshape(-1).reshape(1,-1), value_function_params)
    # 
    # save_bw(state, path=Path(f"exports/{_}.png"))
    states.append(state)
    rewards.append(float(reward))
    if terminated or truncated:
        print(f"GAME {N_GAMES} ENDED: REWARD => {sum(rewards)}")
        gif_path = Path(f"exports/pong/game_{N_GAMES}.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        N_GAMES += 1
        x_train, y_train = generate_value_func_training_examples(rewards, states, 0.95) # x_train = reshaped states, y_train = estimated values
        value_function_params = train(x_train, y_train, value_function_params, HYPERPARAMS, 500)
        observation, info = env.reset()
        frames = []
        states = []
        rewards = []
        print(f"STARTING GAME NUMBER {N_GAMES} [{action_choice_strategy}]")
env.close()