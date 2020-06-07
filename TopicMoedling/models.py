import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import LayerStack


class TopicModeling(nn.Module):
    def __init__(self, topic_k, Doc_num, Word_num):
        super(TopicModeling, self).__init__()
        self.word_num = Word_num

        self.doc_num = Doc_num

        self.Document_Stack = LayerStack(topic_k, Doc_num)
        self.Word_Stack = LayerStack(topic_k, Word_num) 


    def forward(self, type, v, one_hop_list, two_hop_list):
        if type == 'd':
            index = torch.LongTensor([range(self.word_num)]).cuda()
            mat = self.Word_Stack.topic_dist(index).clone()
            x = self.Document_Stack(v, one_hop_list, two_hop_list, mat)

        elif type == 'w':
            index = torch.LongTensor([range(self.doc_num)]).cuda()
            mat = self.Document_Stack.topic_dist(index).clone()
            x = self.Word_Stack(v, one_hop_list, two_hop_list, mat)

        else:
            assert False, 'type error'
        
        return x
            
       