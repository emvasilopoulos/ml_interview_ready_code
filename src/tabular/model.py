import torch.nn as nn


class SimpleDeepClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
