
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, bn, dropout):
        super(Net, self).__init__()

        self.bn = bn
        self.dropout = dropout

        self.linear_1 = nn.Linear(26, 32)
        self.dropout_1 = nn.Dropout(0.5)
        self.bn_1 = nn.BatchNorm1d(32)

        self.linear_2 = nn.Linear(32, 32)
        self.dropout_2 = nn.Dropout(0.5)
        self.bn_2 = nn.BatchNorm1d(32)

        self.linear_3 = nn.Linear(32, 32)
        self.dropout_3 = nn.Dropout(0.5)
        self.bn_3 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 7)
        

    def forward(self, x):
        x = self.linear_1(x)
        if self.bn:
            self.bn_1(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout_1(x)

        x = self.linear_2(x)
        if self.bn:
            self.bn_2(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout_2(x)

        x = self.linear_3(x)
        if self.bn:
            self.bn_3(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout_3(x)
        
        x = self.out(x)

        return x