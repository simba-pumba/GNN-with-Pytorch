import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import LayerStack


class TopicModeling(nn.Module):
    def __init__(self, topic_k, Doc_num, Word_num):
        super(TopicModeling, self).__init__()
        self.Document_Stack = LayerStack(topic_k, Doc_num) 
        self.Word_Stack = LayerStack(topic_k, Word_num) 


    def forward(self, type, v, one_hop_list, two_hop_list):
        if type == 'd':
            mat = copy.deepcopy(self.Word_Stack.topic_dist)
            x = self.Document_Stack(v, two_hop_list, one_hop_list, mat)

        elif type == 'w':
            mat = copy.deepcopy(self.Document_Stack.topic_dist)
            x = self.Word_Stack(v, two_hop_list, one_hop_list, mat)

        else:
            assert False, 'type error'
        
        return x
            
       