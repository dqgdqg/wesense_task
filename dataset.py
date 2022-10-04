
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from IPython import embed
import numpy as np

class MyDataset(Dataset):

    def __init__(self, csv_file):
        self.csv = pd.read_csv(csv_file, index_col=0)
        self.csv = self.csv.reset_index(drop=True)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        label = self.csv['label'][idx]
        data = self.csv.iloc[idx][:-4].to_numpy().astype(np.float32)
        return data, label # 26

if __name__ == '__main__':
    d = MyDataset('./data/train.csv')
    print(d.__getitem__(0))