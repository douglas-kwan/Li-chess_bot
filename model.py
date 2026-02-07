import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  
        return F.relu(out)

class ChessNet(nn.Module):
    def __init__(self, num_move_classes):
        super(ChessNet, self).__init__()

        self.conv_input = nn.Sequential(
            nn.Conv2d(12, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.res_block1 = ResBlock(128)
        self.res_block2 = ResBlock(128)


        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * 8 * 8, num_move_classes)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value