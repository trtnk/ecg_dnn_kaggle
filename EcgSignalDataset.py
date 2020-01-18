from __future__ import print_function, division
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EcgSignalDataset(Dataset):
    def __init__(self, csv_file_path):
        arr = pd.read_csv(csv_file_path, header=None).values
        self.data = arr[:, :-1]
        self.label = arr[:, -1].astype("int")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tmp_data = np.array([self.data[idx]])
        return tmp_data, self.label[idx]
