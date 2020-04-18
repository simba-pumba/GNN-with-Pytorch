"""
layers.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# YJ
class poolAggregateLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation, hidden_dim, bias = True, normalize = True):
        super().__init__()
        self.fc_self = nn.Linear(input_dim, output_dim, bias = False )
        self.activation = activation
       # self.drop = nn.Dropoutd(drop_rate)
        nn.init.xavier_uniform(self.pool, gain=nn.init.calculate_gain(activation)) 
        self.fc_pool = nn.Linear(input_dim, output_dim, bias = bias)
        self.activation = {sigmoid:F.sigmoid, relu:F.relu}[activation]
        self.combine = F.max
        self.norm = F.nomalize


    def forward(self, i, neibs):
        dst = self.fc_self(i)
        dst = self.activation(source)

        src = self.fc_pool(neibs)
        src = self.activation(src)
        
        # if not self.concat:
        #     output = tf.add_n([from_self, from_neighs])
        # else:
        #     output = tf.concat([from_self, from_neighs], axis=1)
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