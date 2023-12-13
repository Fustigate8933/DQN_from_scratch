import torch
from model import DQN
import copy
import cv2
import numpy as np
from collections import deque


class ReplayMemory:
    def __init__(self, replay_capacity, device):
        self.device = device
        self.replay_capacity = replay_capacity

        self.obs = deque(maxlen=replay_capacity)  # deque of tensors
        self.actions = deque(maxlen=replay_capacity)
        self.rewards = deque(maxlen=replay_capacity)
        self.next_obs = deque(maxlen=replay_capacity)
        self.done = deque(maxlen=replay_capacity)

    def sample_batch(self, sample_batch_size):
        sample_indices = np.random.randint(0, self.replay_capacity, (sample_batch_size,), dtype=np.uint8)

        obs_, actions_, rewards_, next_obs_, done_ = (np.array(self.obs), np.array(self.actions), np.array(self.rewards),
                                                      np.array(self.next_obs), np.array(self.done))
        sampled_obs, sampled_actions, sampled_rewards, sampled_next_obs, sampled_done = (obs_[sample_indices],
                                                                                         actions_[sample_indices],
                                                                                         rewards_[sample_indices],
                                                                                         next_obs_[sample_indices],
                                                                                         done_[sample_indices])

        sampled_obs = torch.tensor(sampled_obs, dtype=torch.float32).to(self.device)
        sampled_next_obs = torch.tensor(sampled_next_obs, dtype=torch.float32).to(self.device)

        return sampled_obs, sampled_actions, sampled_rewards, sampled_next_obs, sampled_done

    def add_new_entry(self, obs: torch.Tensor, action: int, reward: float, next_obs: torch.Tensor, done: bool):
        self.obs.append(obs.cpu().numpy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_obs.append(next_obs.cpu().numpy())
        self.done.append(done)

    def __len__(self):
        return len(self.obs)


def epsilon_greedy(epsilon_threshold, actions):
    # threshold decreases over time
    if float(torch.rand(1)) < epsilon_threshold:  # if random number is less than epsilon, take random action
        return int(torch.randint(0, len(actions), (1,)))
    else:
        return int(torch.argmax(actions))


def transform_obs(obs, device):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
    gray = gray[18:102, :] // 255.
    gray = torch.unsqueeze(torch.tensor(gray, dtype=torch.float32).to(device), dim=0)
    return gray


def get_q_values(sampled_rewards, sampled_done, action_values, gamma):
    l = len(sampled_rewards)
    max_action_values = torch.max(action_values, dim=1)[0]
    q_values = torch.zeros(l, dtype=torch.float32)
    for i in range(l):
        if sampled_done[i]:
            q_values[i] = sampled_rewards[i]
        else:
            q_values[i] = sampled_rewards[i] + gamma * max_action_values[i]
    return q_values


def train(env, epochs, replay_capacity, device, sample_batch_size, epsilon=1.0, min_epsilon=0.001, epsilon_decay=0.999, gamma=0.99, seed=42, update_offline_interval=200):
    assert replay_capacity >= sample_batch_size, "Replay capacity for buffer should be >= sample batch size."

    torch.manual_seed(seed)
    np.random.seed(seed)

    action_size = env.action_space.n

    model_online = DQN(action_size).to(device)
    model_offline = copy.deepcopy(model_online)

    model_online.train()

    replay = ReplayMemory(replay_capacity, device)

    optimizer = torch.optim.Adam(model_online.parameters(), lr=1e-3, weight_decay=0)
    mse = torch.nn.MSELoss()

    all_losses, all_rewards = [], []

    for episode in range(epochs):
        obs, _ = env.reset()
        obs = transform_obs(obs, device)  # transforms into normalized, grayscale, 84x84, float32 tensor

        done = False
        action_count = 0
        epoch_reward = 0
        epoch_loss = 0

        while not done:
            action_values = model_online(torch.unsqueeze(obs, dim=0))

            action = epsilon_greedy(epsilon, action_values)
            epsilon = max(min_epsilon, epsilon - epsilon_decay)
            next_obs, reward, done, _, _ = env.step(action)

            epoch_reward += reward
            action_count += 1

            next_obs = transform_obs(next_obs, device)

            replay.add_new_entry(obs, action, reward, next_obs, done)

            if len(replay) == replay_capacity:
                sampled_obs, sampled_actions, sampled_rewards, sampled_next_obs, sampled_done = replay.sample_batch(sample_batch_size)
                action_values = model_offline(sampled_obs)
                q_values = get_q_values(sampled_rewards, sampled_done, action_values, gamma)

                model_offline.eval()
                with torch.no_grad():
                    q_values_pred = torch.max(model_offline(sampled_obs), dim=1)[0].cpu()

                loss = mse(q_values, q_values_pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss

        all_losses.append(epoch_loss)
        all_rewards.append(epoch_reward)

        if (episode + 1) % update_offline_interval == 0:
            model_offline = copy.deepcopy(model_online)
            print(f"Updated offline model at epoch {episode + 1}")

        print(f"Episode #{episode + 1} ended after {action_count} actions. Mean loss: {epoch_loss / action_count:.3f}, Mean reward: {epoch_reward / action_count:.1f}")
