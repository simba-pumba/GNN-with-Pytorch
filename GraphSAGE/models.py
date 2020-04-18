"""
models.py
"""

import torch
import torch.nn as nn
from layers import sagelayer

class GraphSAGE(nn.Module):
    def __init__(self, inputDim, hiddenDim, embedDim, aggregate = "MAX", activation = "sigmoid", drop_rate,featureMatrix):
        """
        Description 
        ...

        """

        super(GraphSAGE, self).__init__()
        self.dimList = [inputDim]
        self.dimList += hiddenDim
        self.dimList += [embedDim]

		numOfLayers = len(self.dimList) - 1
        # layer 다시 쌓기

        #self.drop = nn.Dropout(drop_rate)
        self.layerList = [sagelayer(self.dimList[index], self.dimList[index + 1]) for index in range(numOfLayers)]
        self.featureMatrix = featureMatrix

        # 수정 필요
        self.activation = {sigmoid:F.sigmoid, relu:F.relu}[activation]

        # task 설정
        # pooling layer
        # mlp layer 설정
        self.pooling = ?
        


        self.message_passing_layer = nn.Sequential(*self.layerList)



    def forward(self, vector):
    	# inputFeature = self.featureMatrix[vector]
    	# output = inputFeature

    	# for layer in self.layerList:
    	# 	output = layer(output)

    	return output