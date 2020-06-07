import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler


class GraphDataset(Dataset):
    def __init__(self, G, k = 4, neig_ratio = 1):
        """

        G: Netowrkx object
        neig_ratio
        k: # of negative samples used loss function

        """
        self.k = k
        self.G = G
        self.ratio = neig_ratio
    

    def __len__(self):
        return self.G.number_of_nodes()
        
    def __getitem__(self, index):
        """

        target node의 positive sample

        """

        pos_list = self.pos_samples(index)
                
        two_hop_list = []
        for i in pos_list:
            two_hop_list.append(torch.tensor(self.pos_samples(i)))
        
        """

        loss에 사용되는 positive sample

        """

        pos_loss_node = np.random.choice(pos_list, size = 1).tolist()
        pos_loss_node_one_hop = two_hop_list[pos_list.index(pos_loss_node[0])].tolist()
        pos_loss_node_two_hop = [] # loss용
        for i in pos_loss_node_one_hop:
            pos_loss_node_two_hop.append(torch.tensor(self.pos_samples(i)))
        """

        loss에 사용되는 negative samples

        """
        neg_list = self.neg_samples(index)
        neg_loss_node_one_hop = []
        for i in neg_list:
            neg_loss_node_one_hop.append(torch.tensor(self.pos_samples(i)))
        
        neg_loss_node_two_hop = []
        for i in neg_loss_node_one_hop:
            temp = []
            for j in i.tolist():
                temp.append(torch.tensor(self.pos_samples(j)))         
            neg_loss_node_two_hop.append(temp)
                
        return [index], torch.tensor(pos_list), two_hop_list, pos_loss_node, torch.tensor(pos_loss_node_one_hop), \
               pos_loss_node_two_hop, neg_list, neg_loss_node_one_hop, neg_loss_node_two_hop
        #return [(index, pos_list, two_hop_list), (pos_loss_node, pos_loss_node_one_hop, pos_loss_node_two_hop), (neg_list, neg_loss_node_one_hop, neg_loss_node_two_hop)]



    def neg_samples(self, v):
        neg_neighbor = set(self.G.nodes)-set(self.G.neighbors(v))-set([v])
        neg_neighbor = list(neg_neighbor)
        out_degree = list( map ( lambda a: self.G.degree(a) **(3/4), neg_neighbor ) )
        idx = list(WeightedRandomSampler(out_degree, self.k, replacement=False)) 
        return list(map(lambda i: neg_neighbor[i], idx))

    def pos_samples(self, v):
        temp = list(self.G.adj.get(v))
        num_neig = int( len(temp) * self.ratio  + 1)
        num_neig = min(num_neig, len(temp))
        temp = np.random.choice(temp, size = num_neig, replace = False).tolist()
        return temp


def loss_func(target, pos_loss, neg_set):
        EPS = 1e-15
        pos_loss = torch.dot(target, pos_loss).cuda()
        pos_loss = -torch.log(torch.sigmoid(pos_loss) + EPS).cuda()
        
        neg_loss = torch.zeros(1).cuda()
        for i in neg_set:
            temp = torch.dot(target, i.squeeze().cuda())
            temp = torch.log(torch.sigmoid(-temp) + EPS).cuda()
            neg_loss += temp

        neg_loss = -neg_loss.mean().cuda()

        return pos_loss + neg_loss




# class NegativeSampler:
#     '''
#     v = node index
#     d_v^(3/4), where d_v = the out-degree of vertex v

#     '''
#     def __init__(self, G, k):
#         self.G = G
#         self.k = k
    
#     def neg_sample(self, v):
#         neg_neighbor = set(self.G.nodes)-set(self.G.neighbors(v))-set([v])
#         neg_neighbor = list(neg_neighbor)
#         out_degree = list( map ( lambda a: self.G.degree(a) **(3/4), neg_neighbor ) )
#         idx = list(WeightedRandomSampler(out_degree, self.k, replacement=False))

#         return list(map(lambda i: neg_neighbor[i], idx))
# class PositiveSampler:
#     def __init__(self, G, ratio):
#         self.G = G
#         self.ratio = ratio

#     def neg_sample(self, v):
#         temp = list(self.G.adj.get(v))
#         num_neig = int( len(temp) * self.ratio )
#         temp = np.random.choice(temp, size = num_neig, replace = False).tolist()

#         return list(map(lambda i: neg_neighbor[i], idx))





def calculate_neg_prob(G):
    """
    Calculate the probability distribution for negative sampling.
    """
    degree_list = [math.pow(G.degree[node], 0.75) for node in sorted(G)]

    tot = sum(degree_list)
    degree_list = [value / tot for value in degree_list]

    return degree_list

def get_negative_sample(G, curr, num_of_samples, prob):
    """
    Returns negative samples of specified length.
    """
    neg_nodes = []

    for _ in range(num_of_samples):
        selected_node = np.random.choice(list(G), size = 1, replace = True, p = prob)[0]
        while selected_node in G.neighbors(curr) or selected_node == curr:
            selected_node = np.random.choice(list(G), size = 1, replace = True, p = prob)[0]
        neg_nodes.append(selected_node)
    return neg_nodes


class node_Dataset_ns(Dataset):
    def __init__(self, G, table, negative_ratio):
        self.G = G # g: networkx.Graph()
        self.table = table # Noise distribution for nodes

        pos_edge = []
        for edge in G.edges:
            pos_edge.append([edge[0], edge[1]])
            pos_edge.append([edge[1], edge[0]])

        pos_edge = np.array(pos_edge)

        self.neg_table = np.zeros((2*len(G.edges), negative_ratio))

        for index, row in enumerate(pos_edge):
            neg_list = get_negative_sample(G, row[0], negative_ratio, self.table)
            self.neg_table[index] = np.array(neg_list)

        self.target = np.array(pos_edge[:,0])
        self.context = np.array(pos_edge[:,1])
        self.negative = np.array(self.neg_table, dtype = int)

    def __getitem__(self, index):
        return [self.target[index], self.context[index], self.negative[index]] # long tensor 수정 필

    def __len__(self):
        return len(self.target)

