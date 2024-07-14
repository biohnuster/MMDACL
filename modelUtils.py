import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score,  precision_recall_curve
from sklearn.metrics import auc as auc3
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from sklearn.metrics.pairwise import cosine_similarity as cos
import time
import scipy.spatial.distance as dist
import copy
import random
import pdb
import warnings
warnings.filterwarnings("ignore")


def load_graph(feature_edges, n):
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sparse.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n),
                             dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sparse.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nfadj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def construct_fgraph(features, topk):
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edge = []
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                edge.append([i, vv])
    return edge

def generate_knn(data):
    topk = 3

    edge = construct_fgraph(data, topk)
    res = []

    for line in edge:
        start, end = line[0], line[1]
        if int(start) < int(end):
            res.append([start, end])
    return res

def aug_random_mask(input_feature, drop_percent=0.4):
    # input_feature = input_feature.detach()
    input_feature = torch.tensor(input_feature)
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    mask_idx = random.sample(node_idx, mask_num)

    for i in range(input_feature.shape[0]):
        # mask_idx = random.sample(node_idx, mask_num)

        for j in mask_idx:
            aug_feature[i][j] = zeros
    return aug_feature

def aug_random_edge(input_adj, drop_percent=0.4):

    percent = drop_percent

    edge_num = len(input_adj)  # 9228 / 2
    add_drop_num = int(edge_num * percent)
    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)
    drop_idx.sort()
    drop_idx.reverse()
    for i in drop_idx:
        input_adj = np.delete(input_adj, i, axis=0)
    return input_adj

def constructure_graph(dateset, h1, h2, task="dti", aug=False):

    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)
    edge = np.loadtxt(f"{task}edge.txt", dtype=int)

    if aug:
        edge_aug = aug_random_edge(np.array(edge))
        edge_aug = load_graph(np.array(edge_aug), dateset.shape[0])
        edge = load_graph(np.array(edge), dateset.shape[0])

        feature_aug = aug_random_mask(feature)
        return edge, feature, edge_aug, feature_aug
    edge = load_graph(np.array(edge), dateset.shape[0])

    return edge, feature

def constructure_knngraph(dateset, h1, h2, aug=False):

    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)
    fedge = np.array(generate_knn(feature.cpu().detach().numpy()))

    if aug:
        fedge_aug = aug_random_edge(np.array(fedge))
        feature_aug = aug_random_mask(feature)
        fedge_aug = load_graph(np.array(fedge_aug), dateset.shape[0])
        fedge = load_graph(np.array(fedge), dateset.shape[0])

        return fedge, feature, fedge_aug, feature_aug
    else:
        fedge = load_graph(np.array(fedge), dateset.shape[0])

        return fedge, feature

def constructure_catfeature(dateset, h1, h2):

    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)

    return feature

def get_clGraph(data, task):
    cledg = np.loadtxt(f"{task}_cledge.txt", dtype=int)

    cl = torch.eye(len(data))
    for i in cledg:
        cl[i[0]][i[1]] = 1
    return cl

def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg

