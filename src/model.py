import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    

    def __init__(self):

        super(ImageClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(in_features=32*32*32, out_features=128)

        self.fc2 = nn.Linear(in_features=128, out_features=3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x

