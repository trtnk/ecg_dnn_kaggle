# -*- coding: utf-8 -*-
"""
@file execute_test.py
@version 1.0
@author trtnk
@date 2020/01/18
@brief Execute ecg dnn test
@details Use MIT-BIH dataset
@warning
@note
"""

import argparse

import torch
import torch.utils.data as data
from sklearn.metrics import classification_report, confusion_matrix

from EcgSignalDataset import EcgSignalDataset
from EcgSignalCNN import EcgSignalCNN, EcgSignalCNN2
import network_tools

# create parser
parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
parser.add_argument("model_path")
parser.add_argument("-b", "--batch", type=int, default=500)
args = parser.parse_args()

csv_file_path = args.dataset_path
model_path = args.model_path
batch_size = args.batch

# Read dataset
file_base_name = "mitbih"
test_dataset = EcgSignalDataset(f"{csv_file_path}/{file_base_name}_test.csv")

# Create Dataloader
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# load trained model
#net = EcgSignalCNN()
net = EcgSignalCNN2()
net.load_state_dict(torch.load(model_path))

# test
## whole accuracy
predicted_labels, correct_labels = network_tools.test_model(net, test_dataloader)
class_names = ['N', 'S', 'V', 'F', 'Q']
print(classification_report(correct_labels, predicted_labels, target_names=class_names))
print(confusion_matrix(correct_labels, predicted_labels))
## each classes accuracy
network_tools.test_model_each_class(net, test_dataloader, 5)
