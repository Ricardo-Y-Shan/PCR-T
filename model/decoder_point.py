import torch
import torch.nn as nn


class DecoderPoint(nn.Module):
    def __init__(self, input_dim=1024, n_points=1024):
        super().__init__()
        self.hidden_dim = 1024
        self.n_points = n_points

        self.relu = nn.ReLU()
        self.fc_0 = nn.Linear(input_dim, self.hidden_dim)
        self.fc_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, n_points * 3)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.fc_0(x)
        x = self.fc_1(self.relu(x))
        x = self.fc_2(self.relu(x))
        points = self.fc_out(self.relu(x))
        points = points.view(batch_size, -1, 3)

        return points
