import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils import NegativeSampler 

class Attention(nn.Module):
    def __init__(self, n_topics):
        super(Attention, self).__init__()

        # self.norm = nn.BatchNorm1d(n_topics)

        self.conv_d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1,
                      padding=0, bias=False)
        self.conv_W = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1,
                      padding=0, bias=False)




    def forward(self, x, word_matrix):

        #x_norm = self.norm(x)
        d = self.conv_d(x)
        W = self.conv_W(word_matrix)

        dW_T = torch.matmul(d, W.transpose(1,2))    # output shape [batch, 1, |W|]

        att = F.softmax(dW_T * word_matrix, dim=1)  # output shape [batch, 1, n_topics]
        output = x * att                            # output shape [batch, 1, n_topics]

        return output



class Conv(nn.Module):
    def __init__(self, topic_k):
        super(Conv, self).__init__()
        self.conv = nn.Linear(topic_k, topic_k)
        self.self_conv = nn.Linear(topic_k, topic_k, bias = False)
    
    def forward(self, v_feature, pos_feature_list):
        neighbor_mean = torch.mean(pos_feature_list, dim = pos_feature_list.size(-1))
        out = self.conv(neighbor_mean)
        x = self.self_conv(v_feature)
        x = x + out
        x = F.relu(x)  
        x = F.normalize(x, dim = -1)
        return x



class LayerStack(nn.Module):
    def __init__(self, topic_k, Doc_num, dropout = 0.25):
        super(LayerStack, self).__init__()
        self.topic_dist = nn.Embedding(num_embeddings = Doc_num, embedding_dim = topic_k)
        nn.init.xavier_uniform_(self.topic_dist.weight, gain=1.0)


        #self.neg_sampling_func = NegativeSampler
        self.two_hop_conv = Conv(topic_k) 
        self.one_hop_conv = Conv(topic_k) 
        self.Attention = Attention(topic_k)
    

    def forward(self, v, one_hop_list, two_hop_list, W_mat):

        """
        
        v: vertex
        pos_list: positive neighbor sampling
        neg_list: negative negibhor sampling

        """
        v = v[0]
        v.to('cuda')
        #_, one_hop_list = one_hop_list
        one_hop_list = torch.tensor(one_hop_list[0])
        one_hop_list.to('cuda')
        #_, two_hop_list = two_hop_list
        W_mat.to('cuda')
        x = self.topic_dist(v)
        one_hop_features = self.topic_dist(one_hop_list)
        # two_hop_list = two_hop_list[0]

        for index, tens in enumerate(one_hop_features):
            neighbor_two_hop = torch.tensor(two_hop_list[index])
            neighbor_two_hop.to('cuda')
            pos_feature_list = self.topic_dist(neighbor_two_hop)
            one_hop_features[index] = self.two_hop_conv(tens, pos_feature_list)

        x = self.one_hop_conv(x, one_hop_features)  # x = self.Doc_topic_dist(x), squeeze(1) 
        x = self.Attention(x, W_mat)
        x = F.normalize(x)

        return x 
