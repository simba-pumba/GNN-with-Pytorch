import warnings
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
import pandas as pd

from utils import GraphDataset, loss_func
from models import TopicModeling

warnings.simplefilter("ignore", UserWarning)

#if __name__ == "__main__":
# 2000
# 13000

def main(documnet_graph, word_graph):
    # documnet_graph = nx.karate_club_graph()
    # word_graph = nx.karate_club_graph()
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
    word_dataloader = DataLoader(dataset2, batch_size = 1 , shuffle = True)


    model = TopicModeling(topic_k, Doc_num, Word_num)
    # model = nn.DataParallel(model)
    model.cuda()

    opt = optim.Adam(model.parameters(), lr = learning_rate)



    for epoch in tqdm(range(epochs)):
        model.train()
        print("Document: ", epoch)
        loss = torch.tensor(0.0).cuda()
        for i, data in enumerate(doc_dataloader):
            """
            d
            """
            # unpacking

            index, pos_list, two_hop_list, \
                pos_loss_node, pos_loss_node_one_hop, pos_loss_node_two_hop, \
                    neg_list, neg_loss_node_one_hop, neg_loss_node_two_hop = data

            target = model(type = 'd', v = index[0],one_hop_list= pos_list.squeeze(), two_hop_list = two_hop_list)
            pos_loss = model('d', pos_loss_node[0],  pos_loss_node_one_hop.squeeze(), pos_loss_node_two_hop)
            neg_loss = []
            for k, node_i in enumerate(neg_list):
                neg_loss.append(model('d', node_i, neg_loss_node_one_hop[k].squeeze(), \
                    neg_loss_node_two_hop[k]))

            loss = loss + loss_func(target.squeeze().cuda(), pos_loss.squeeze().cuda(), \
                neg_loss).cuda()
            if (i % doc_batch_size == 0):
                loss.cuda()
                loss.backward()
                opt.step()
                print(i,"batch 완료")
                print("loss: ", loss)
                loss = torch.tensor(0.0).cuda()
                
        loss = torch.tensor(0.0).cuda()
        print("Word: ", epoch)
        model.train()
        for i, data in enumerate(word_dataloader):
            """
            w
            """
            index, pos_list, two_hop_list, \
                pos_loss_node, pos_loss_node_one_hop, pos_loss_node_two_hop, \
                    neg_list, neg_loss_node_one_hop, neg_loss_node_two_hop = data
            target = model('w', index[0], pos_list.squeeze(), two_hop_list)
            pos_loss = model('w', pos_loss_node[0],  pos_loss_node_one_hop.squeeze(), pos_loss_node_two_hop)
            neg_loss = []
            for k, node_i in enumerate(neg_list):
                    neg_loss.append(model('w', node_i, neg_loss_node_one_hop[k].squeeze(), neg_loss_node_two_hop[k]).squeeze())

            loss = loss + loss_func(target.squeeze().cuda(), pos_loss.squeeze().cuda(), neg_loss).cuda()
            if (i % word_batch_size == 0):
                loss.backward()
                opt.step()
                print(i,"batch 완료")
                print("loss: ", loss)
                loss = torch.tensor(0.0).cuda()

        torch.save(model.state_dict(), "./results/model_"+str(epoch)+".pth")
        print(epoch, ", 모델 저장 완료")


if __name__ == '__main__':
    doc_graph = pd.read_csv("d2v_lambda_0.14034.csv", usecols=['Y_gene', 'X_gene'])
    doc_graph = doc_graph.rename(columns={'Y_gene' : 'source', 'X_gene' : 'target'})
    doc_graph = nx.from_pandas_edgelist(doc_graph)
    word_graph = pd.read_csv("w2v_lambda_8_0.156.csv", usecols=['Y_gene', 'X_gene'])
    word_graph = word_graph.rename(columns={'Y_gene' : 'source', 'X_gene' : 'target'})
    word_graph = nx.from_pandas_edgelist(word_graph)

    main(doc_graph, word_graph)