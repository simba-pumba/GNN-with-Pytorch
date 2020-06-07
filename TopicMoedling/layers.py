import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils import NegativeSampler 

class Attention(nn.Module):
    def __init__(self, topic_k):
        super(Attention, self).__init__()

        # self.norm = nn.BatchNorm1d(n_topics)

        # self.conv_d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1,
        #               padding=0, bias=False)
        # self.conv_W = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1,
        #               padding=0, bias=False)

        self.fc_d = nn.Linear(topic_k, topic_k, bias=False)
        self.fc_W = nn.Linear(topic_k, topic_k, bias=False)




    def forward(self, x, word_matrix):
        word_matrix = word_matrix.squeeze()
        #x_norm = self.norm(x)
        d = self.fc_d(x)
        W = self.fc_W(word_matrix)

        dW_T = torch.matmul(d, W.transpose(0, 1))    # output shape [batch, 1, |W|]
        att = F.softmax(torch.matmul(dW_T, word_matrix))  # output shape [batch, 1, n_topics]
        output = x * att                            # output shape [batch, 1, n_topics]

        return output



class Conv(nn.Module):
    def __init__(self, topic_k):
        super(Conv, self).__init__()
        self.conv = nn.Linear(topic_k, topic_k)
        self.self_conv = nn.Linear(topic_k, topic_k, bias = False)
    
    def forward(self, v_feature, pos_feature_list):
        
        if pos_feature_list.dim != 1:
            neighbor_mean = torch.mean(pos_feature_list, dim = 0)
        else:
            neighbor_mean = pos_feature_list

        out = self.conv(neighbor_mean)
        x = self.self_conv(v_feature)
        x = x + out
        x = F.relu(x)  
        x = F.normalize(x, dim = -1)
        return x



class LayerStack(nn.Module):
    def __init__(self, topic_k, Doc_num, dropout = 0.25):
        super(LayerStack, self).__init__()
        self.topic_dist = nn.Embedding(num_embeddings = Doc_num, embedding_dim = topic_k).cuda()
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
        # v = v[0]
        # v.cuda()
        #_, one_hop_list = one_hop_list
        # one_hop_list = torch.tensor(one_hop_list[0])
        #one_hop_list.cuda()
        #_, two_hop_list = two_hop_list
        #W_mat.cuda()
        x = self.topic_dist(v.long().cuda())
        one_hop_features = self.topic_dist(one_hop_list.long().cuda())
        # two_hop_list = two_hop_list[0]

        if type(two_hop_list) != list:
            two_hop_list = [two_hop_list]
            # print('######## [two_hop_list]', two_hop_list)
        if one_hop_features.dim() == 1:
            one_hop_features = one_hop_features.unsqueeze(0)
        out = one_hop_features.clone()
        for index, tens in enumerate(one_hop_features):
            # print(one_hop_features.shape, len(two_hop_list))
            neighbor_two_hop = two_hop_list[index].squeeze()
            neighbor_two_hop.cuda()
            pos_feature_list = self.topic_dist(neighbor_two_hop.long().cuda())
            if pos_feature_list.dim() == 1:
                pos_feature_list = pos_feature_list.unsqueeze(0)
            out[index] = self.two_hop_conv(tens, pos_feature_list)

        x = self.one_hop_conv(x, out)  # x = self.Doc_topic_dist(x), squeeze(1) 
        x = self.Attention(x, W_mat.cuda())
        x = F.normalize(x)

        return x 
