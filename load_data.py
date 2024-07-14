import numpy as np
import torch
import torch as th
import dgl
import warnings
import pandas as pd
import pickle

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

def loadmeidata():
    # Reconstruct Drug-Drug interaction network
    # 1 interaction + 2 sim
    drug_drug = pd.read_csv('dataset/meidata/drug_drug.csv')  # drug-drug关联相似度
    # drug_sim = pd.read_csv('./dataset/meidata/1/mat_drug_drug.csv', header=None).values

    # Reconstruct Protein-Protein interaxtion network
    # 1 interaction + 2 sim
    protein_protein = pd.read_csv('dataset/meidata/protein_protein.csv')  # protein_protein关联相似度
    # protein_sim = pd.read_csv('./dataset/meidata/1/mat_protein_protein.csv', header=None).values

    drug_chemical = pd.read_csv('dataset/meidata/drug_chemical.csv')
    protein_sequence = pd.read_csv('dataset/meidata/protein_sequence.csv')

    protein_disease = pd.read_csv('dataset/meidata/protein_disease.csv')

    drug_protein = pd.read_csv('dataset/meidata/drug_protein.csv')

    drug_se = pd.read_csv('dataset/meidata/drug_se.csv')
    drug_disease = pd.read_csv('dataset/meidata/drug_disease.csv')

    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_chemical', 'drug'): (th.tensor(drug_chemical['Drug'].values),
                                            th.tensor(drug_chemical['Chemical'].values)),

        ('drug', 'chemical_drug', 'drug'): (th.tensor(drug_chemical['Chemical'].values),
                                            th.tensor(drug_chemical['Drug'].values)),

        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),

        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('protein', 'protein_sequence', 'protein'): (th.tensor(protein_sequence['Protein'].values),
                                                     th.tensor(protein_sequence['Sequence'].values)),

        ('protein', 'sequence_protein', 'protein'): (th.tensor(protein_sequence['Sequence'].values),
                                                     th.tensor(protein_sequence['Protein'].values)),

        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),

        ('protein', 'protein_disease', 'disease'): (th.tensor(protein_disease['Protein'].values),
                                                    th.tensor(protein_disease['Disease'].values)),
        ('disease', 'disease_protein', 'protein'): (th.tensor(protein_disease['Disease'].values),
                                                    th.tensor(protein_disease['Protein'].values)),

        ('drug', 'drug_se', 'se'): (th.tensor(drug_se['Drug'].values),
                                    th.tensor(drug_se['Se'].values)),
        ('se', 'se_drug', 'drug'): (th.tensor(drug_se['Se'].values),
                                    th.tensor(drug_se['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)

    # 使用随机数作为节点特征
    drug_feature = np.hstack((torch.randn((g.num_nodes('drug'), g.num_nodes('drug'))), np.zeros((g.num_nodes('drug'), g.num_nodes('protein')))))
    protein_feature = np.hstack((np.zeros((g.num_nodes('protein'), g.num_nodes('drug'))), torch.randn((g.num_nodes('protein'), g.num_nodes('protein')))))

    # 使用one-hot编码作为节点特征
    # drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('protein')))))
    # protein_feature = np.hstack((np.zeros((g.num_nodes('protein'), g.num_nodes('drug'))), protein_sim))

    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.from_numpy(protein_feature).to(th.float32)
    return g
def loadluodata():
    # Reconstruct Drug-Drug interaction network
    # 1 interaction + 2 sim
    drug_drug = pd.read_csv('dataset/luodata/drug_drug.csv')  # drug-drug关联相似度
    # drug_sim = pd.read_csv('./dataset/luodata/Sim_mat_drug_drug.csv', header=None).values

    # Reconstruct Protein-Protein interaxtion network
    # 1 interaction + 2 sim
    protein_protein = pd.read_csv('dataset/luodata/protein_protein.csv')  # protein_protein关联相似度
    # protein_sim = pd.read_csv('./dataset/luodata/Sim_mat_protein_protein.csv', header=None).values

    drug_chemical = pd.read_csv('dataset/luodata/drug_chemical.csv')
    protein_sequence = pd.read_csv('dataset/luodata/protein_sequence.csv')

    protein_disease = pd.read_csv('./dataset/luodata/protein_disease.csv')

    drug_protein = pd.read_csv('./dataset/luodata/drug_protein.csv')

    drug_se = pd.read_csv('./dataset/luodata/drug_se.csv')
    drug_disease = pd.read_csv('./dataset/luodata/drug_disease.csv')

    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),

        ('drug', 'drug_chemical', 'drug'): (th.tensor(drug_chemical['Drug'].values),
                                            th.tensor(drug_chemical['Chemical'].values)),
        ('drug', 'chemical_drug', 'drug'): (th.tensor(drug_chemical['Chemical'].values),
                                            th.tensor(drug_chemical['Drug'].values)),

        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),

        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),

        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),

        ('protein', 'protein_sequence', 'protein'): (th.tensor(protein_sequence['Protein'].values),
                                                     th.tensor(protein_sequence['Sequence'].values)),
        ('protein', 'sequence_protein', 'protein'): (th.tensor(protein_sequence['Sequence'].values),
                                                     th.tensor(protein_sequence['Protein'].values)),

        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),

        ('protein', 'protein_disease', 'disease'): (th.tensor(protein_disease['Protein'].values),
                                                    th.tensor(protein_disease['Disease'].values)),
        ('disease', 'disease_protein', 'protein'): (th.tensor(protein_disease['Disease'].values),
                                                    th.tensor(protein_disease['Protein'].values)),

        ('drug', 'drug_se', 'se'): (th.tensor(drug_se['Drug'].values),
                                    th.tensor(drug_se['Se'].values)),
        ('se', 'se_drug', 'drug'): (th.tensor(drug_se['Se'].values),
                                    th.tensor(drug_se['Drug'].values)),

        # ('protein', 'protein_gene', 'gene'): (th.tensor(protein_gene['Protein'].values),
        #                                       th.tensor(protein_gene['Gene'].values)),
        # ('gene', 'gene_protein', 'protein'): (th.tensor(protein_gene['Gene'].values),
        #                                       th.tensor(protein_gene['Protein'].values)),
        # ('gene', 'gene_gene', 'gene'): (th.tensor(gene_gene['Gene1'].values),
        #                                 th.tensor(gene_gene['Gene2'].values)),
        # ('gene', 'gene_pathway', 'pathway'): (th.tensor(gene_pathway['Gene'].values),
        #                                       th.tensor(gene_pathway['Pathway'].values)),
        # ('pathway', 'pathway_gene', 'gene'): (th.tensor(gene_pathway['Pathway'].values),
        #                                       th.tensor(gene_pathway['Gene'].values)),
        # ('pathway', 'pathway_pathway', 'pathway'): (th.tensor(pathway_pathway['Pathway1'].values),
        #                                             th.tensor(pathway_pathway['Pathway2'].values)),
        # ('pathway', 'pathway_disease', 'disease'): (th.tensor(pathway_disease['Pathway'].values),
        #                                             th.tensor(pathway_disease['Disease'].values)),
        # ('disease', 'disease_pathway', 'pathway'): (th.tensor(pathway_disease['Disease'].values),
        #                                             th.tensor(pathway_disease['Pathway'].values)),
        # ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
        #                                             th.tensor(disease_disease['Disease2'].values)),
    }
    g = dgl.heterograph(graph_data)

    # 使用随机数作为节点特征
    drug_feature = np.hstack((torch.randn((g.num_nodes('drug'), g.num_nodes('drug'))), np.zeros((g.num_nodes('drug'), g.num_nodes('protein')))))
    protein_feature = np.hstack((np.zeros((g.num_nodes('protein'), g.num_nodes('drug'))), torch.randn((g.num_nodes('protein'), g.num_nodes('protein')))))

    # 使用化学相似性作为节点特征
    # drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('protein')))))
    # protein_feature = np.hstack((np.zeros((g.num_nodes('protein'), g.num_nodes('drug'))), protein_sim))

    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.from_numpy(protein_feature).to(th.float32)
    return g

def loadzhengdata():
    # Reconstruct Drug-Drug interaction network
    drug_drug = pd.read_csv('dataset/zhengdata/drug_drug.csv')  # drug-drug关联相似度
    # Reconstruct Protein-Protein interaxtion network
    protein_protein = pd.read_csv('dataset/zhengdata/target_target.csv')  # protein_protein关联相似度

    drug_su = pd.read_csv('dataset/zhengdata/drug_su.csv')
    drug_s = pd.read_csv('./dataset/zhengdata/drug_s.csv')
    drug_st = pd.read_csv('./dataset/zhengdata/drug_st.csv')
    protein_GO = pd.read_csv('./dataset/zhengdata/target_GO.csv')

    drug_protein = pd.read_csv('./dataset/zhengdata/drug_target.csv')

    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),

        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),

        ('drug', 'drug_su', 'su'): (th.tensor(drug_su['Drug'].values),
                                        th.tensor(drug_su['Su'].values)),
        ('su', 'su_drug', 'drug'): (th.tensor(drug_su['Su'].values),
                                      th.tensor(drug_su['Drug'].values)),

        ('drug', 'drug_s', 's'): (th.tensor(drug_s['Drug'].values),
                                        th.tensor(drug_s['S'].values)),
        ('s', 's_drug', 'drug'): (th.tensor(drug_s['S'].values),
                                     th.tensor(drug_s['Drug'].values)),

        ('drug', 'drug_st', 'st'): (th.tensor(drug_st['Drug'].values),
                                        th.tensor(drug_st['St'].values)),
        ('st', 'st_drug', 'drug'): (th.tensor(drug_st['St'].values),
                                      th.tensor(drug_st['Drug'].values)),

        ('protein', 'protein_GO', 'GO'): (th.tensor(protein_GO['Protein'].values),
                                          th.tensor(protein_GO['GO'].values)),
        ('GO', 'GO_protein', 'protein'): (th.tensor(protein_GO['GO'].values),
                                          th.tensor(protein_GO['Protein'].values)),

        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)

    # 使用随机数作为节点特征
    drug_feature = np.hstack((torch.randn((g.num_nodes('drug'), g.num_nodes('drug'))), np.zeros((g.num_nodes('drug'), g.num_nodes('protein')))))
    protein_feature = np.hstack((np.zeros((g.num_nodes('protein'), g.num_nodes('drug'))), torch.randn((g.num_nodes('protein'), g.num_nodes('protein')))))

    # 使用化学相似性作为节点特征
    # drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('protein')))))
    # protein_feature = np.hstack((np.zeros((g.num_nodes('protein'), g.num_nodes('drug'))), protein_sim))

    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.from_numpy(protein_feature).to(th.float32)
    return g
def   remove_graph(g, test_id):
    """Delete the drug-disease associations which belong to test set
    from heterogeneous network.
    """

    test_drug_id = test_id[:, 0]
    # print('test_drug_id:', test_drug_id)

    test_protein_id = test_id[:, 1]
    edges_id = g.edge_ids(th.from_numpy(np.array(test_drug_id)),
                          th.from_numpy(np.array(test_protein_id)),
                          etype=('drug', 'drug_protein', 'protein'))
    g = dgl.remove_edges(g, edges_id, etype=('drug', 'drug_protein', 'protein'))
    edges_id = g.edge_ids(th.tensor(test_protein_id),
                          th.tensor(test_drug_id),
                          etype=('protein', 'protein_drug', 'drug'))
    g = dgl.remove_edges(g, edges_id, etype=('protein', 'protein_drug', 'drug'))
    return g

