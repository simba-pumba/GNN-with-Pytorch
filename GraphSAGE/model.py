import utils
import layer
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, layer_dim, aggregate = "MAX", activation = "sigmoid", supervised = False, n_classes = 0):
        '''
        layer_dim: {1: (input_dim, output_dim), 2: (hidden_dim,output_dim) , ..., L: (hidden_dim, output_dim)}
        

        '''
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.Activation = {sigmoid:F.sigmoid, relu:F.relu}[activation]
        
        self.layer = []
        for i in layer_dim:
            input_dim = i[0]
            output_dim = i[1]
            layer.append(layer.Aggregate(input_dim, output_dim))
            layer.append(self.Activation)
        
        self.layer = nn.Sequential(layer_dim)

        if supervised:
            self.fc = nn.Linear(output_dim, n_classes)



    def forward(self):

        



