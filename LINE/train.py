#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn, optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
import networkx as nx
import numpy as np

import model
import visualization


# In[ ]:


class TrainLINE:
    def __init__(self, G, emd_dim, epochs,learning_rate = 0.01, neg_k = 2, second_order = False):
        '''
        G: networkx 객체
        emd_dim: embedding dimension
        neg_k: # negative samples
        batch: edges
        '''
        self.line = LINE(G = G,emd_dim = emd_dim, neg_k = 2, second_order = second_order)
        self.learning_rate = learning_rate
    
        self.batch = list(G.edges)
        self.nodes = len(G.nodes)
        self.epochs = epochs
    
    def train(self):
        opt = optim.SGD(self.line.parameters(), lr=self.learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.line.cuda()
        
        for epoch in range(self.epochs):
            self.line.train()
            
            # batch로 쓰일 edges를 epoch마다 random하게 섞어줌
            np.random.shuffle(self.batch) 
            
            print("Epoch {}".format(epoch))
            
            for b in tqdm(self.batch):
                self.line.zero_grad()
                loss = self.line(i=torch.tensor(b[0],device=device), j=torch.tensor(b[1],device=device), device=device)
                loss.backward()
                opt.step()
                
            # epoch마다 embedding vectors의 plot을 뽑음.    
            self.line.eval()
            Visualization().TSNE(self.line.emd_layer(torch.tensor(range(self.nodes),device=device)).cpu().detach().numpy(),epoch)
            

        print("\nDone\n")        
        


# In[ ]:


if __name__ == "__main__":
    model = TrainLINE(nx.karate_club_graph(), epochs = 100, learning_rate = 0.0001, emd_dim = 5, neg_k=2, second_order = True)
    model.train()

