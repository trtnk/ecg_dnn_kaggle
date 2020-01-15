from EcgSignalCNN import EcgSignalCNN

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = EcgSignalCNN().to(device)

batch_size = 500
x_check = torch.FloatTensor(batch_size, 1, 187)
x_check = x_check.to(device)

net.check_cnn_size(x_check)