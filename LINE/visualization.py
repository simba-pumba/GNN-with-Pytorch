#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE


# In[ ]:


class Visualization:
    def __init__(self, labels = None):
        # karate graph label 임시로 고정
        if labels is None:
            self.labels = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
    def TSNE(self, matrix,iter_num):
        model = TSNE(n_components=2, random_state=0)
        model.fit_transform(matrix)
        sns.scatterplot(matrix[:,0],matrix[:,1],hue = self.labels)
        plt.savefig('fig'+str(iter_num)+'.png', dpi=300)
        plt.close()

