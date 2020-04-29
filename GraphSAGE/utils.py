"""
utils.py
"""
from torch.utils.data import DataLoader
from torch.utils.data import Datast
from torchvision import transforms
from torch.nn as nn
from torch.nn.functional as F
import networkx as nx
import numpy as np


def preprocess_cora():
    raw_graph = nx.read_edgelist("cora.cites", nodetype = int)
    connected_comp = sorted(nx.connected_components(raw_graph), key=len, reverse=True)
    # Get LCC(Largest Connected Component) of Cora
    lcc_graph = raw_graph.subgraph(connected_comp[0])
    # Node indices of LCC
    lcc_nodelist = sorted(lcc_graph)
    raw_features = np.genfromtxt("cora.content", dtype = np.dtype(str))
    # Select node features contained in LCC
    lcc_index = np.array([i for i, row in enumerate(raw_features) if int(row[0]) in lcc_nodelist])
    lcc_features = raw_features[lcc_index]
    # Get the binary feature matrix
    feature = np.array(lcc_features[:, 1:-1], dtype = np.float32)

    # Transform labels (string) into integers
    class_set = set(lcc_features[:,-1])
    classes_dict_int = {c: i for i, c in enumerate(classes)}
    label = np.array(list(map(classes_dict_int.get, labels)), dtype = np.int32)

    # Re-index nodes
    index = sorted(set(np.array(lcc_features[:,0], dtype = int)))
    index_map = {j:i for i,j in enumerate(index)}
    final_graph = nx.relabel_nodes(lcc_graph, index_map)
    adjacency = nx.adjacency_matrix(final_graph)
    
    return adjacency, feature, index_map, label

def batch_generator(graph, target_index, k_hop):
  pass


class BatchGenerator(Dataset):
      def __init__(self, Adjacency, Feature, index_map, label, k_hop):
        self.A = Adjacency
        self.G = nx.from_numpy_matrix(Adjacency)
        self.F = Feature #numpy matrix
        self.index_map = index_map
        self.label = label
        self.k = k_hop
        self.num_nodes = self.A.shape[0]
        
        #self.idx2idx = dict(zip(??? , range(num_nodes)))

      
      
      def __len__(self):
        return self.num_nodes


      def __getitem__(self, index):
        k_hop_neig= nx.single_source_shortest_path_length(self.G, index, cutoff=self.k).keys()
        num_sub_nodes = len(k_hop_neig)
        subgraph = G.subgraph(k_hop_neig)
        sub_A = nx.adjacency_matrix(subgraph).toarray()
        sub2global = dict(zip(range(len(num_sub_nodes)),subgraph.nodes))
        sub_F = self.F[list(k)]
        return (torch.DoubleTensor(index),torch.DoubleTensor(torch.sub_A), torch.FloatTensor(sub_F), torch.DoubleTensor(label))
