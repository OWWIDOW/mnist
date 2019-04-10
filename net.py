import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=5),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(32*400, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = x.view(-1,32*400)
        # print(x.size())
        x = self.fc1(x)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return F.log_softmax(x, dim=1)
