"""
layers.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# YJ
class poolAggregateLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation, bias = True, normalize = True):
        super().__init__()
        self.fc_self = nn.Linear(input_dim, output_dim, bias = False)
       # self.drop = nn.Dropoutd(drop_rate)
        nn.init.xavier_uniform(self.fc_self, gain=nn.init.calculate_gain(activation)) 

    
        self.fc_pool = nn.Linear(input_dim, output_dim, bias = bias)
        nn.init.xavier_uniform(self.pool, gain=nn.init.calculate_gain(activation)) 
        self.activation = {'sigmoid':F.sigmoid,'relu':F.relu}[activation]
        self.combine = F.max
        self.norm = F.nomalize


    def forward(self, i, neibs):
        dst = self.fc_self(i)
        dst = self.activation(dst)
 
        src = self.fc_pool(neibs)
        src = self.activation(src)
        
        out = self.combine( dst + src )
        out = self.norm(out)
        return out 
    	

# YM
class minAggregateLayer(nn.Module):
    def __init__():
        super(minAggregateLayer, self).__init__()
        pass

    def forward(self):
    	pass

# KJ
class lstmAggregateLayer(nn.Module):
    def __init__():
        super(lstmAggregateLayer, self).__init__()
        pass

    def forward(self):
    	pass

class combinelayer(nn.Module):
    def __init__():
        super(combinelayer, self).__init__()
        pass

    def forward(self):
    	pass