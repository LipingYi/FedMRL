from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import numpy as np
import math
import torch


########################## FedMRL ##########################
class CNN_1_large(nn.Module): # for hetero. exp.
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_1_large, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = F.relu(self.fc2(x))
        x = self.fc3(m_rep_large)
        return x, hetero_rep




class CNN_1_small(nn.Module): # for hetero. exp.
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10,small_rep_dim=250):
        super(CNN_1_small, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 2000)
        self.fc2 = nn.Linear(2000, small_rep_dim)
        self.fc3 = nn.Linear(small_rep_dim, out_dim)

    def forward(self, x, m_rep_small):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        homo_rep = F.relu(self.fc2(x))
        x = self.fc3(m_rep_small)
        return x, homo_rep


class CNN_5_small(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10,small_rep_dim=250):
        super(CNN_5_small, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 500)
        self.fc2 = nn.Linear(500, small_rep_dim)
        self.fc3 = nn.Linear(small_rep_dim, out_dim)

    def forward(self, x, m_rep_small):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        homo_rep = F.relu(self.fc2(x))
        x = self.fc3(m_rep_small)
        return x, homo_rep

class projector(nn.Module):
    def __init__(self, in_dim=750, out_dim=500):
        super(projector, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 400)
        # self.fc2 = nn.Linear(400, 200)
        # self.fc3 = nn.Linear(200, out_dim)
        # self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 2000)
        # self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(in_dim, out_dim)
        # self.fc4 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # o = F.relu(self.fc2(x))
        o = self.fc3(x)
        # o = self.fc4(o)
        return o


class CNN_2_large(nn.Module): # change filters of convs
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_2_large, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, n_kernels, 5)
        self.fc1 = nn.Linear(n_kernels * 5 * 5, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = F.relu(self.fc2(x))
        x = self.fc3(m_rep_large)
        return x, hetero_rep

class CNN_3_large(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_3_large, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = F.relu(self.fc2(x))
        x = self.fc3(m_rep_large)
        return x, hetero_rep


class CNN_4_large(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_4_large, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = F.relu(self.fc2(x))
        x = self.fc3(m_rep_large)
        return x, hetero_rep

class CNN_5_large(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_large, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, m_rep_large):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        hetero_rep = F.relu(self.fc2(x))
        x = self.fc3(m_rep_large)
        return x, hetero_rep


########################## FedMRL ##########################