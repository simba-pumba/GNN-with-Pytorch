#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx
import numpy as np

import graph_data


# In[ ]:


class LINE(nn.Module):
    def __init__(self, G, emd_dim, neg_k = 2, second_order = False):
        '''
        
        emd_dim: embedding dimension
        neg_k: # negative samples  
        
        
        '''
        super(LINE, self).__init__() 
        self.G = GraphData(G) 
        self.V = len(G.nodes) # number of nodes
        self.neg_k = neg_k 
        
        self.emd_dim = emd_dim
        
        self.emd_layer = nn.Embedding(num_embeddings = self.V, 
                               embedding_dim = emd_dim)
        
        self.second_order = second_order
        if self.second_order:  # second_order를 위한 context_layer 생성
            self.context_layer = nn.Embedding(num_embeddings = self.V, 
                               embedding_dim = emd_dim)
            
   
    def forward(self, i, j, device):
        v_i = self.emd_layer(i).to(device) # target node의 embeddding vector
        if self.second_order:  # first neighbor의 embeddding vector
            v_j = self.context_layer(j).to(device) # second_order일 경우, context layer에서 가져옴.
        else: # first neighbor의 embeddding vector
            v_j = self.emd_layer(j).to(device)
        
        # negative sampling의 postive 파트
        pos_loss = F.logsigmoid(torch.matmul(v_i, v_j)).to(device) 
        
        # negative sampling의 negative 파트
        neg_set = self.G.NegativeSampler(v = i.tolist() , k = self.neg_k) # v: target node, k: # of negative samples
        neg_set = torch.tensor(neg_set).to(device)
        neg_loss = 0
        
        for v_n in neg_set:
            if self.second_order:
                prob = torch.matmul(v_i,self.context_layer(v_n)).to(device)
                neg_loss += F.logsigmoid(-prob).to(device)
                
            else: # first_order
                prob = torch.matmul(v_i,self.emd_layer(v_n)).to(device)
                neg_loss += F.logsigmoid(-prob).to(device)
                
        negative_sampling = pos_loss + neg_loss
        
        return -negative_sampling
        

