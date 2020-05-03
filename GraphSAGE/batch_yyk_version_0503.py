
import networkx as nx
import random
class BatchGenerator(Dataset):
      def __init__(self, Adjacency, Feature, index_map, label, k_hop, sample_rate = 1):
        self.A = Adjacency
        self.G = nx.from_numpy_matrix(Adjacency)
        self.F = Feature #numpy matrix
        self.index_map = index_map
        self.label = label
        self.k = k_hop
        self.num_nodes = self.A.shape[0]
        self.sample_rate = sample_rate



    
      def __len__(self):
        return self.num_nodes



      def __getitem__(self, index):
        # 함수로 빼야할듯!!!!! 
        # 코드 길다 주륵
        computation_list = []
        currlayer = [index]
        nodeset = set(target)
        for k in range(self.k):
          nextlayer = []
          for nodelist in currlayer:
            for node in nodelist:
              neigh = list(self.A.neighbors(node))

              if index in neigh:
                neigh.remove(index)
              num_neigb = max(1, int(len(neigh)*self.sample_rate)-1)

              neigh = random.shuffle(neigh)[:num_neigb]
              nextlayer.append(neigh)
              nodeset.add(node)
          computation_list.append(nextlayer)


        num_sub_nodes = len(nodeset)
        subgraph = G.subgraph(nodeset)
        sub_A = nx.adjacency_matrix(subgraph).toarray()
        sub2global = dict(zip(range(len(num_sub_nodes)),subgraph.nodes))
        sub_F = self.F[list(subgraph.nodes)]
        return (torch.DoubleTensor(index),torch.DoubleTensor(torch.sub_A), torch.FloatTensor(sub_F), torch.DoubleTensor(computation_list))
