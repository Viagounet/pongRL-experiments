import ale_py
import shimmy
import numpy as np
import gymnasium as gym

# 1. Création de l'environnement
# 'render_mode' permet de voir le jeu s'afficher à l'écran
env = gym.make("ALE/Pong-v5")

# 2. Réinitialisation de l'environnement
observation, info = env.reset()

def merge_frames(frames: list[np.array]) -> np.array:
    return np.array(frames).mean(axis=0)

frames = []
for _ in range(1000):
    # 3. Choisir une action aléatoire (0-5 dans Pong)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(observation)
    print(merge_frames(frames).shape)
    if terminated or truncated:
        observation, info = env.reset()

env.close()