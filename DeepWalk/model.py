import numpy as np
import torch
from torch.autograd import Variable as Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import networkx as nx


def H_Softmax_index(u_k, num_V):
    with torch.no_grad():
        u=u_k.tolist()
        last_parent_index = num_V-2
        new_u_k = last_parent_index + u + 1  # u_k는 0-base 이므로
        update_parents = []
        while (new_u_k >= 0):
            if new_u_k % 2 == 0:
                new_u_k = (int(new_u_k)-1) / int(2)
            else:
                new_u_k = int(new_u_k) / int(2)
            update_parents.append(new_u_k)
            if new_u_k == 0:
                break
        return torch.tensor(update_parents,dtype=torch.long)


class DeepWalk(nn.Module):
    def __init__(self, emd_dim, num_V):
        super(DeepWalk, self).__init__()
        self.emd_dim = emd_dim
        self.num_V = num_V
        self.emd_layer = nn.Embedding(num_embeddings = num_V, embedding_dim = emd_dim).to(torch.float64)
        self.emd_layer.weight.data.uniform_(0, 0.3)
        self.h_softmax = nn.Embedding(num_embeddings = num_V-1, embedding_dim = emd_dim).to(torch.float64) # n개의 leaf 노드를 갖기 위해서는 n-1개의 부모들이 존재해야함.
        self.h_softmax.weight.data.uniform_(0, 0.3)
        self.Sigmoid = nn.functional.logsigmoid 
    def forward(self, v_j, u_k, device):
        emd = self.emd_layer(v_j.to(torch.long)).to(device)
        updates = H_Softmax_index(u_k.to(torch.long), self.num_V).squeeze().to(device)
        log_pr = torch.tensor(1).to(torch.float64).to(device)
        for i in updates:
            log_pr *= self.Sigmoid(torch.dot(emd.to(torch.float64),self.h_softmax(i).to(torch.float64)).to(torch.float64)).to(torch.float64)
        
        return -log_pr
        