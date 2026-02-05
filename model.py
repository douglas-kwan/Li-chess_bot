import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Backbone: Input is 12 planes of 8x8
        self.conv_block = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Policy Head (Classification)
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672) # 4672 possible moves
        )
        
        # Value Head (Regression)
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 1),
            nn.Tanh() # Scales output between -1 and 1
        )

    def forward(self, x):
        x = self.conv_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value