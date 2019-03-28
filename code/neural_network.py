import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.add_module('fc1', nn.Linear(6, 45))
        self.layers.add_module('fc2', nn.Linear(45, 45))
        self.layers.add_module('fc3', nn.Linear(45, 20))
        self.layers.add_module('fc4', nn.Linear(20, 1))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < (len(self.layers) - 1):
                x = F.relu(layer(x))
                continue
            x = layer(x)
        return x


    part_layer_3 = {}
    part_layer_4 = {}