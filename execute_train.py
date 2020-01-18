# -*- coding: utf-8 -*-
"""
@file execute_train.py
@version 1.0
@author trtnk
@date 2020/01/18
@brief Execute ecg dnn train and validation
@details Use MIT-BIH dataset
@warning
@note
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from EcgSignalDataset import EcgSignalDataset
from EcgSignalCNN import EcgSignalCNN
import network_tools

# create parser
parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("-b", "--batch", type=int, default=500)
parser.add_argument("-e", "--epoch", type=int, default=2)
parser.add_argument("-s", "--save_dir", default='.')
args = parser.parse_args()

csv_file_path = args.path
batch_size = args.batch
num_epochs = args.epoch
save_dir = args.save_dir

# Read dataset
file_base_name = "mitbih"
train_val_dataset = EcgSignalDataset(f"{csv_file_path}/{file_base_name}_train.csv")
val_size = 10000
train_size = int(len(train_val_dataset) - val_size)
train_dataset, val_dataset = data.random_split(train_val_dataset, [train_size, val_size])
test_dataset = EcgSignalDataset(f"{csv_file_path}/{file_base_name}_test.csv")

# Create Dataloader
dataloader_dict = {
    "train": data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    "val": data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    "test": data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
}

# network
net = EcgSignalCNN()

# define loss function
criterion = nn.CrossEntropyLoss()

# define optimization method
optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

# training and validation
network_tools.train_model(net, dataloader_dict, criterion, optimizer, num_epochs)

# save trained model
save_path = f"{save_dir}/epoch{num_epochs}_weights_ecg_cnn.model"
torch.save(net.state_dict(), save_path)
