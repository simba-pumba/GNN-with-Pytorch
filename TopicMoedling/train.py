import numpy as np
import torch
from torch.autograd import Variable as Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import networkx as nx
from tqdm import tqdm

from utils import GraphDataset, loss_func
from models import TopicModeling

#if __name__ == "__main__":
# 2000
# 13000
documnet_graph = nx.karate_club_graph()
word_graph = nx.karate_club_graph()
topic_k = 4
k = 4 # # of negative samples    
Doc_num = documnet_graph.number_of_nodes()
Word_num = word_graph.number_of_nodes()
# Doc_NegativeSample_func = NegativeSampler(documnet_graph, k = 4)
# Word_NegativeSample_func = NegativeSampler(word_graph, k = 4)

epochs = 1000
doc_batch_size = 64
# word_batch_size = int(Word_num / (float(Doc_num) / batch_size))
word_batch_size = 128
neig_ratio = 0.8
device = torch.device('cuda')
learning_rate = 1e-4

dataset1 = GraphDataset(documnet_graph, k = k, neig_ratio = neig_ratio)
dataset2 = GraphDataset(word_graph, k = k, neig_ratio = neig_ratio)
doc_dataloader = DataLoader(dataset1, batch_size = 1, shuffle = True)
word_dataloader = DataLoader(dataset2, batch_size = word_batch_size , shuffle = True)


model = TopicModeling(topic_k, Doc_num, Word_num)
# model = nn.DataParallel(model)
model.cuda()

opt = optim.Adam(model.parameters(), lr = learning_rate)



for epoch in range(epochs):
    for data in tqdm(doc_dataloader):
        """
        d
        """
        # unpacking
        index, pos_list, two_hop_list, \
            pos_loss_node, pos_loss_node_one_hop, pos_loss_node_two_hop, \
                neg_list, neg_loss_node_one_hop, neg_loss_node_two_hop = data 
        neg_loss_node_two_hop = neg_loss_node_two_hop[0]
        target = model('d', index, pos_list, two_hop_list)
        pos_loss = model('d', pos_loss_node,  pos_loss_node_one_hop, pos_loss_node_two_hop)
        neg_loss = []
        for k, node_i in enumerate(neg_list):
                neg_loss.append(model('d', node_i, neg_loss_node_one_hop[k], neg_loss_node_two_hop[k][k]))
        
        loss = loss_func(target, pos_loss, neg_loss)
        
        loss.backward()
        opt.step()
    
    print("Document: ", epoch)
    print("loss: ", loss)

    for data in tqdm(word_dataloader):
        """
        w
        """
        # unpacking
        
        index, pos_list, two_hop_list, \
            pos_loss_node, pos_loss_node_one_hop, pos_loss_node_two_hop, \
                neg_list, neg_loss_node_one_hop, neg_loss_node_two_hop = data 

        target = model('w', index, pos_list, two_hop_list)
        pos_loss = model('w', pos_loss_node,  pos_loss_node_one_hop, pos_loss_node_two_hop)
        neg_loss = []
        for k, node_i in enumerate(neg_list):
                neg_loss.append(model('w', node_i, neg_loss_node_one_hop[k], neg_loss_node_two_hop[k][k]))
        
        loss = loss_func(target, pos_loss, neg_loss)
        
        loss.backward()
        opt.step()
    print("Word: ", epoch)
    print("loss: ", loss)








    




