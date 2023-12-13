import gymnasium as gym
import pygame
import torch
from train import train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

env = gym.make('Pong-v4', render_mode='rgb_array')
obs, info = env.reset()

train(env, 500, 5, device, 5, 1, 0.001, 0.999, 0.99, 42, 10)


env.step(0)
pygame.quit()
