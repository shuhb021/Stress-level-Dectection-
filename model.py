import torch
import torch.nn as nn

class EmotionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,7)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x