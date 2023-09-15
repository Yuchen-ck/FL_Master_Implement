from torch import nn
from torch.nn import functional as F


class MNIST_Net_paper(nn.Module):
    def __init__(self):
        super(MNIST_Net_paper, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(25600, 128)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = x.view(-1, self.num_flat_features(x))
        #         print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
