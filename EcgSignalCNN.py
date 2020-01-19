import torch.nn as nn
import torch.nn.functional as F

class EcgSignalCNN(nn.Module):
    def __init__(self):
        super(EcgSignalCNN, self).__init__()
        # input is 1ch ecg signal
        # convolution layers
        self.conv = nn.Conv1d(1, 32, 5)

        self.conv11 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv12 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv21 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv22 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv31 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv32 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv41 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv42 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv51 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv52 = nn.Conv1d(32, 32, 5, padding=2)

        self.pool = nn.MaxPool1d(5, stride=2)
        self.fc1 = nn.Linear(32*2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        x_tmp = self.conv(x)
        x = F.relu(self.conv11(x_tmp))
        x = F.relu(self.conv12(x) + x_tmp)
        x_tmp = self.pool(x)
        x = F.relu(self.conv21(x_tmp))
        x = F.relu(self.conv22(x) + x_tmp)
        x_tmp = self.pool(x)
        x = F.relu(self.conv31(x_tmp))
        x = F.relu(self.conv32(x) + x_tmp)
        x_tmp = self.pool(x)
        x = F.relu(self.conv41(x_tmp))
        x = F.relu(self.conv42(x) + x_tmp)
        x_tmp = self.pool(x)
        x = F.relu(self.conv51(x_tmp))
        x = F.relu(self.conv52(x) + x_tmp)
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def check_cnn_size(self, x):
        x_tmp = self.conv(x)
        print(x_tmp.size())
        x = F.relu(self.conv11(x_tmp))
        print(x.size())
        x = F.relu(self.conv12(x) + x_tmp)
        print(x.size())
        x_tmp = self.pool(x)
        print(x_tmp.size())
        x = F.relu(self.conv21(x_tmp))
        print(x.size())
        x = F.relu(self.conv22(x) + x_tmp)
        print(x.size())
        x_tmp = self.pool(x)
        print(x_tmp.size())
        x = F.relu(self.conv31(x_tmp))
        print(x.size())
        x = F.relu(self.conv32(x) + x_tmp)
        print(x.size())
        x_tmp = self.pool(x)
        print(x_tmp.size())
        x = F.relu(self.conv41(x_tmp))
        print(x.size())
        x = F.relu(self.conv42(x) + x_tmp)
        print(x.size())
        x_tmp = self.pool(x)
        print(x_tmp.size())
        x = F.relu(self.conv51(x_tmp))
        print(x.size())
        x = F.relu(self.conv52(x) + x_tmp)
        print(x.size())
        x = self.pool(x)
        print(x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Reduced a CNN layer version
class EcgSignalCNN2(nn.Module):
    def __init__(self):
        super(EcgSignalCNN2, self).__init__()
        # input is 1ch ecg signal
        # convolution layers
        self.conv = nn.Conv1d(1, 32, 5)

        self.conv11 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv12 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv21 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv22 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv31 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv32 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv41 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv42 = nn.Conv1d(32, 32, 5, padding=2)

        self.pool = nn.MaxPool1d(5, stride=2)
        self.fc1 = nn.Linear(32*8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        x_tmp = self.conv(x)
        x = F.relu(self.conv11(x_tmp))
        x = F.relu(self.conv12(x) + x_tmp)
        x_tmp = self.pool(x)
        x = F.relu(self.conv21(x_tmp))
        x = F.relu(self.conv22(x) + x_tmp)
        x_tmp = self.pool(x)
        x = F.relu(self.conv31(x_tmp))
        x = F.relu(self.conv32(x) + x_tmp)
        x_tmp = self.pool(x)
        x = F.relu(self.conv41(x_tmp))
        x = F.relu(self.conv42(x) + x_tmp)
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
