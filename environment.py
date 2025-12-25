import ale_py
import shimmy
import numpy as np
import gymnasium as gym

from pathlib import Path
from state_utils import frames_to_bw_resized
from image_utils import save_bw, save_rgb
env = gym.make("ALE/Pong-v5")

# 2. Réinitialisation de l'environnement
observation, info = env.reset()
frames = []
for _ in range(1000):
    # 3. Choisir une action aléatoire (0-5 dans Pong)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(observation)
    state = frames_to_bw_resized(frames, 4, (105, 80))
    save_bw(state, path=Path(f"exports/{_}.png"))
    if terminated or truncated:
        observation, info = env.reset()

env.close()