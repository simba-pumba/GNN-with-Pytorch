#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
from torch.utils.data.sampler import WeightedRandomSampler


# In[2]:


class GraphData:
    def __init__(self, G):
        
        '''
        G: networkx 객체
        G.nodes, G.edges, G.neighbors(i) 
        '''
        self.G=G
        
    
    def NegativeSampler(self, v, k=2 ,with_replacement=False):
        '''
        k: sample의 개수
        i: vertex
        d_v^(3/4), where d_v = the out-degree of vertex v
        '''
        
        # v의 negative nodes set 구하기 
        neg_neighbor = set(self.G.nodes)-set(self.G.neighbors(v))-set([v])
        
        neg_neighbor = list(neg_neighbor)
        
        # out degree^(3/4) 구하기 
        out_degree = list( map ( lambda a: self.G.degree(a) **(3/4), neg_neighbor ) )
        
        # out_degree 분포를 고려하여 k개의 랜덤 샘플을 뽑음
        idx = list(WeightedRandomSampler(out_degree, k, replacement=with_replacement))
            
        
        return list(map(lambda i: neg_neighbor[i], idx))

