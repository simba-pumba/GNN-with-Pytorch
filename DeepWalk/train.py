
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import numpy as np
import torch
from torch.autograd import Variable as Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import networkx as nx
from . import model, Visualization



class GraphDataset(Dataset):
    def __init__(self, G, t, w, transform = None):
        self.adj=G.adj
        self.num = G.number_of_nodes()
        self.index = list(G.nodes())
        self.transform = transform
        self.get_neighbors = self.adj.get
        self.t = t
        self.w = w
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, index):
        # randomwalk
        RandomWalk = [index]
        for i in range(1, self.t):
            neighbors = list(self.get_neighbors(RandomWalk[i-1]))
            RandomWalk.append(np.random.choice(neighbors))       
        return (torch.tensor(RandomWalk,dtype=int))



class trainDW:
    def __init__(self, epochs, t, w, emd_dim, num_V, train_loader, lr):
        self.epochs = epochs
        self.model = DeepWalk(emd_dim, num_V)
        self.train_loader = train_loader
        self.w = w
        self.t = t 
        self.lr = lr
        self.num_V = num_V
        
        
    def window_neighbor(self, RW, index):
        u_k_set = []
        lst = RW.tolist()
        for i in np.arange(-self.w, self.w+1):
            if i == 0:
                continue
            try:
                u_k_set.append(lst[index+i])
            except IndexError:
                continue
        return torch.tensor(u_k_set)
        
        
    def train(self):
        device = torch.device("cpu")
        opt = optim.SGD(self.model.parameters(), lr=self.lr)
#         self.model.cuda()
        
        for epoch in range(self.epochs):
            
            self.model.train()
            for RW in tqdm(self.train_loader):
                RW = RW.squeeze() 
                for index, v_j in enumerate(RW):
                    u_k_set = self.window_neighbor(RW, index).squeeze()
                    
                    for u_k in u_k_set:
                        self.model.zero_grad()
                        loss = self.model(v_j.to(torch.long).to(device), u_k.to(torch.long).to(device), device)
                        loss.backward()
                        opt.step()
            
            self.model.eval()
            Visualization.Visualization.TSNE(self.model.emd_layer(torch.tensor(range(self.num_V))).cpu().detach().numpy(),epoch)
            

if __name__ == '__main__':
    '''args'''
    t = 10
    w = 2
    epochs = 100
    lr = 0.001
    emd_dim = 6
    num_V = 34

    trainset = GraphDataset(nx.karate_club_graph(), t, w)
    train_loader = DataLoader(trainset, batch_size = 1, shuffle = True ) # epoch마다 shuffle
    model = trainDW(epochs, t, w, emd_dim, num_V, train_loader, lr)
    model.train()