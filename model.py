import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        # input shape is [None, 1, 84, 84]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding="valid")
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding="valid")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.f = nn.Flatten()
        self.dense = nn.Linear(in_features=64*9*9, out_features=action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(f"Input x shape: {x.shape}")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
#         print(f"Shape before flatten: {x.shape}")
        x = self.f(x)
#         print(f"Shape after flatten: {x.shape}")
        x = self.dense(x)
#         print(f"Output x shape: {x.shape}")
        return x
