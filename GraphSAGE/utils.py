"""
utils.py
"""

from scipy import sparse as sp
import numpy as np

def preprocess_cora(content_path, cites_path):
	"""
	"""
	cora_content = np.loadtxt(path, dtype = str)
	index = {j: i for i,j in enumerate(np.array(cora_content[:, 0], dtype = int))}
	label = np.array(cora_content[:,-1])
	feature = sp.csr_matrix(cora_content[:,1: -1].astype(float))

	cora_cites = np.loadtxt(cites_path, dtype = int)
	adjacency = np.zeros([len(index),len(index)])
	for i in cora_cites:
    		adjacency[i[0],i[1]]=1
			adjacency[i[1],i[0]]=1

	sp.csr_matrix = sp.csr_matrix(adjacency)
	return adjacency, feature, index, label

