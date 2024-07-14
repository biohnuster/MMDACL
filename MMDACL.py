import torch
from sklearn.preprocessing import normalize
from scipy import sparse as sp
from imports import *
import numpy as np
from numpy import ndarray
import pickle
from modelUtils import *
import dgl
from dgl.nn.pytorch import GraphConv
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 规范化
def normalize_matrix(mat: sp.csr_matrix) -> sp.csr_matrix:
    normalized_mat = normalize(mat)
    return normalized_mat

#  hg: dgl.DGLHeteroGraph, metapath: list[str], feat_dict: dict[NodeType, FloatTensor]
def aggregate_metapath_neighbors(hg, node_feature_att, metapath, feat_dict) -> FloatArray:
    etypes = set(hg.canonical_etypes)  # 返回图中的所有规范边类型,eg:{（'drug','drug-protein','drug）,.....}
    etype_map = {etype[1]: etype for etype in
                 etypes}  # 建立边map，eg:{'drug-protein':（'drug','drug-protein','drug）,.....}
    src_ntype = etype_map[metapath[0]][0]  # 一条元路径的开始节点
    dest_ntype = etype_map[metapath[-1]][2]  # 一条元路径的终止节点
    assert src_ntype == dest_ntype  # 开始节点和终止节点必须统一
    feat = feat_dict[src_ntype].cpu().numpy()  # 加载节点的特征信息

    product = None

    # 对本条元路径进行节点乘法操作和特征聚合
    for etype in metapath:
        '''
        etype:('drug', 'drug_protein', 'protein'),...
        '''
        etype = etype_map[etype]

        adj_mat = hg.adj_external(etype=etype, scipy_fmt='csr').astype(np.float32)  # 提取图中每步元路径关系的邻接矩阵
        normalized_adj_mat = normalize_matrix(adj_mat)
        # print('normalized_adj_mat:',normalized_adj_mat)

        if product is None:
            product = normalized_adj_mat
        else:
            # 关系矩阵乘法
            product = product.dot(normalized_adj_mat)

    # 特征聚合
    # sub_g=dgl.heterograph

    node_feature_att = node_feature_att.cpu().detach().numpy()
    out = product.dot(node_feature_att * feat)
    assert isinstance(out, ndarray) and out.dtype == np.float32 and out.shape == feat.shape

    return out


# hg: dgl.DGLHeteroGraph,infer_ntype: NodeType,feat_dict: dict[NodeType, FloatTensor],metapath_list: list[list[str]]
def pre_aggregate_neighbor(aggr_feat_list, metapath_node_feature_att_matrix, hg, infer_ntype, feat_dict, metapath_list):
    aggr_feat_list.clear()

    # 将未经聚合的初始特征加入
    raw_feat = feat_dict[infer_ntype].cpu().numpy()
    aggr_feat_list.append(raw_feat)

    for metapath, node_feature_att in zip(metapath_list, metapath_node_feature_att_matrix):
        aggr_feat = aggregate_metapath_neighbors(
            hg=hg,
            node_feature_att=node_feature_att,
            metapath=metapath,
            feat_dict=feat_dict,
        )
        aggr_feat_list.append(aggr_feat)
    return aggr_feat_list

class GraphConvolution(Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nfeat//2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        res = x2
        return res

class CL_GCN(nn.Module):
    def __init__(self, nfeat, dropout, alpha = 0.8):
        super(CL_GCN, self).__init__()
        self.gcn1 = GCN(nfeat, dropout)
        self.gcn2 = GCN(nfeat, dropout)
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2, clm):
        z1 = self.gcn1(x1, adj1)
        z2 = self.gcn2(x2, adj2)
        cl_loss = self.sim(z1, z2, clm)
        return z1, z2, cl_loss

    def sim(self, z1, z2, clm):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        sim_matrix = sim_matrix.to(device)

        loss = -torch.log(sim_matrix.mul(clm).sum(dim=-1)).mean()
        return loss

    def mix2(self, z1, z2):
        loss = ((z1 - z2) ** 2).sum() / z1.shape[0]
        return loss

class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(32, 2, bias=False),)
            # nn.LogSoftmax(dim=1),
            # nn.Sigmoid())
    def forward(self, x):
        output = self.MLP(x)
        return output

class Metapath_Embeding_fun(nn.Module):
    '''
    Decoder model for drug-protein association prediction
    '''

    def __init__(self, input_feature_dim, num_metapaths, fea_dropout=0.4):
        super(Metapath_Embeding_fun, self).__init__()
        # self.inner_feature_dim = inner_feature_dim
        self.input_feature_dim = input_feature_dim
        self.num_metapaths = num_metapaths
        self.dropout = nn.Dropout(fea_dropout)

        self.weight_attn = nn.Parameter(torch.Tensor(self.num_metapaths, 1, 1))
        nn.init.normal_(self.weight_attn)

        self.weights_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.input_feature_dim, self.input_feature_dim)) for _ in
             range(self.num_metapaths)])

    def forward(self, drug_fea_tensor, protein_fea_tensor):
        drug_fea_list = [drug_fea for drug_fea, weight in
                               zip(drug_fea_tensor, self.weights_list)]
        drug_fea_list = torch.stack(drug_fea_list)
        drug_fea = torch.sum(drug_fea_list * (F.softmax(self.weight_attn, dim=0)), dim=0)

        protein_fea_list = [protein_fea for protein_fea, weight in
                               zip(protein_fea_tensor, self.weights_list)]
        protein_fea_list = torch.stack(protein_fea_list)
        protein_fea = torch.sum(protein_fea_list * (F.softmax(self.weight_attn, dim=0)), dim=0)

        return drug_fea, protein_fea


class MMDACL_bio(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_metapaths: int):
        super(MMDACL_bio, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = 0.4
        self.feature_node_name = ['drug', 'protein']

        # 结点未经聚合的原始特征，作为一个特殊的元路径聚合结果
        self.num_metapaths = num_metapaths + 1

        # node feature attention params matrix
        self.drug_metapath_node_feature_att_matrix = nn.Parameter(torch.ones(self.num_metapaths, 708, 1))
        nn.init.uniform_(self.drug_metapath_node_feature_att_matrix)
        self.protein_metapath_node_feature_att_matrix = nn.Parameter(torch.ones(self.num_metapaths, 1512, 1))
        nn.init.uniform_(self.protein_metapath_node_feature_att_matrix)

        # collection meta_path feature aggr list
        self.aggr_drug_feat_list = []
        self.aggr_protein_feat_list = []

        # MLP for drug feature_projection
        self.drug_feature_projector_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((in_dim + hidden_dim) // 2, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((in_dim + hidden_dim) // 2, hidden_dim),
            )
            for _ in range(self.num_metapaths)
        ])
        # MLP for protein feature_projection
        self.protein_feature_projector_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((in_dim + hidden_dim) // 2, (in_dim + hidden_dim) // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear((in_dim + hidden_dim) // 2, hidden_dim),
            )
            for _ in range(self.num_metapaths)
        ])

        # transformer for feature semantic aggregation
        self.Q_drug = nn.Linear(hidden_dim, hidden_dim)
        self.K_drug = nn.Linear(hidden_dim, hidden_dim)
        self.V_drug = nn.Linear(hidden_dim, hidden_dim)
        self.beta_drug = Parameter(torch.ones(1))

        self.Q_protein = nn.Linear(hidden_dim, hidden_dim)
        self.K_protein = nn.Linear(hidden_dim, hidden_dim)
        self.V_protein = nn.Linear(hidden_dim, hidden_dim)
        self.beta_protein = Parameter(torch.ones(1))

        # graph contrastive learning DTI prediction
        self.CL_GCN = CL_GCN(hidden_dim*2, self.dropout)
        self.MLP = MLP(hidden_dim*2)
        self.metapathsEmbeding_fun = Metapath_Embeding_fun(self.hidden_dim, self.num_metapaths)
        self.loss = nn.CrossEntropyLoss()
        # self.prediction_fun = Multiplex_InnerPrgunoductDecoder(self.hidden_dim, self.num_metapaths)

    def forward(self, g, feature, metapath_list, cl, dateset_index, data, label, iftrain=True, attn_out_drug=None, attn_out_protein=None) -> FloatTensor:
        # 1. Simplified Neighbor Aggregation ,cl, dateset_index, data, iftrain=True, attn_out_drug=None, attn_out_protein=None
        # for drug feature aggregation
        if iftrain:
            self.aggr_drug_feat_list = pre_aggregate_neighbor(self.aggr_drug_feat_list,
                                                              self.drug_metapath_node_feature_att_matrix, g,
                                                              self.feature_node_name[0],
                                                              feature,
                                                              metapath_list[0])

            # for protein feature aggregation
            self.aggr_protein_feat_list = pre_aggregate_neighbor(self.aggr_protein_feat_list,
                                                                 self.protein_metapath_node_feature_att_matrix, g,
                                                                 self.feature_node_name[1],
                                                                 feature,
                                                                 metapath_list[1])

            assert len(self.aggr_drug_feat_list) == self.num_metapaths and len(
                self.aggr_protein_feat_list) == self.num_metapaths

            drug_num_nodes, protein_num_nodes = len(self.aggr_drug_feat_list[0]), len(self.aggr_protein_feat_list[0])

            # 2. Multi-layer Feature Projection
            assert len(self.aggr_drug_feat_list) == len(self.drug_feature_projector_list) and len(
                self.aggr_protein_feat_list) == len(self.protein_feature_projector_list)

            drug_proj_list = [
                proj(
                    torch.from_numpy(feat).cuda()
                )
                for feat, proj in zip(self.aggr_drug_feat_list, self.drug_feature_projector_list)
            ]

            protein_proj_list = [
                proj(
                    torch.from_numpy(feat).cuda()
                )
                for feat, proj in zip(self.aggr_protein_feat_list, self.protein_feature_projector_list)
            ]
            # Turn into tensor
            drug_proj = torch.stack(drug_proj_list)
            protein_proj = torch.stack(protein_proj_list)

            assert drug_proj.shape == (self.num_metapaths, drug_num_nodes, self.hidden_dim) and protein_proj.shape == (
                self.num_metapaths, protein_num_nodes, self.hidden_dim)

            drug_proj = drug_proj.transpose(0, 1)
            protein_proj = protein_proj.transpose(0, 1)

            assert drug_proj.shape == (drug_num_nodes, self.num_metapaths, self.hidden_dim) and protein_proj.shape == (
                protein_num_nodes, self.num_metapaths, self.hidden_dim)

            # 3. Transformer-based Semantic Aggregation
            Q_drug = self.Q_drug(drug_proj)
            K_drug = self.K_drug(drug_proj)
            V_drug = self.V_drug(drug_proj)
            assert Q_drug.shape == K_drug.shape == V_drug.shape == (drug_num_nodes, self.num_metapaths, self.hidden_dim)

            Q_protein = self.Q_protein(protein_proj)
            K_protein = self.K_protein(protein_proj)
            V_protein = self.V_protein(protein_proj)
            assert Q_protein.shape == K_protein.shape == V_protein.shape == (
                protein_num_nodes, self.num_metapaths, self.hidden_dim)

            # for drug transformer operation
            attn_drug = Q_drug @ (K_drug.transpose(1, 2))
            assert attn_drug.shape == (drug_num_nodes, self.num_metapaths, self.num_metapaths)

            attn_drug = torch.softmax(attn_drug, dim=-1)

            attn_out_drug = self.beta_drug * (attn_drug @ drug_proj) + drug_proj
            assert attn_out_drug.shape == (drug_num_nodes, self.num_metapaths, self.hidden_dim)

            attn_out_drug = attn_out_drug.view(self.num_metapaths, drug_num_nodes, self.hidden_dim)

            # for protein transformer operation
            attn_protein = Q_protein @ (K_protein.transpose(1, 2))
            assert attn_protein.shape == (protein_num_nodes, self.num_metapaths, self.num_metapaths)

            attn_protein = torch.softmax(attn_protein, dim=-1)

            attn_out_protein = self.beta_protein * (attn_protein @ protein_proj) + protein_proj
            assert attn_out_protein.shape == (protein_num_nodes, self.num_metapaths, self.hidden_dim)

            attn_out_protein = attn_out_protein.view(self.num_metapaths, protein_num_nodes, self.hidden_dim)

            attn_out_drug, attn_out_protein = self.metapathsEmbeding_fun(attn_out_drug, attn_out_protein)

        # for graph contrastive learning DTI prediction
        edge, feature  = constructure_graph(data, attn_out_drug, attn_out_protein)# 构造拓扑图
        f_edge, f_feature = constructure_knngraph(data, attn_out_drug, attn_out_protein)  # 构造语义图
        feature1, feature2, cl_loss1 = self.CL_GCN(feature, edge, f_feature, f_edge, cl)
        out_prediction = self.MLP(torch.cat((feature1, feature2), dim=1)[dateset_index])
        mlp_loss = self.loss(out_prediction, label.squeeze(1).long())
        out_prediction = out_prediction.sigmoid()

        if iftrain:
            return out_prediction, mlp_loss+cl_loss1, attn_out_drug, attn_out_protein
        return out_prediction

def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)

