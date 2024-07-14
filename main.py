from imports import *
import numpy as np
import pandas as pd
import torch as th
from MMDACL import *
from warnings import simplefilter
from sklearn.model_selection import StratifiedKFold
from load_data import loadmeidata, remove_graph, loadluodata, loadzhengdata
from evaluationUtils import get_metrics_auc, set_seed, plot_result_auc, \
    plot_result_aupr, EarlyStopping, get_metrics
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# General Arguments
parser.add_argument('-id', '--device_id', default='0', type=str,
                    help='Set the device (GPU ids).')
# parser.add_argument('-da', '--dataset', default='KFCdataset_baseline', type=str,
#                     help='Set the data set for training.')
parser.add_argument('-sp', '--saved_path', type=str,
                    help='Path to save training results', default='result')
parser.add_argument('-se', '--seed', default=51000, type=int,
                    help='Global random seed')

# Training Arguments
parser.add_argument('-fo', '--nfold', default=5, type=int,
                    help='The number of k in K-folds Validation')
parser.add_argument('-ep', '--epoch', default=500, type=int,
                    help='Number of epochs for training')
parser.add_argument('-lr', '--learning_rate', default=0.005, type=float,
                    help='learning rate to use')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to use')
parser.add_argument('-pa', '--patience', default=100, type=int,
                    help='Early Stopping argument')
# Model Arguments
parser.add_argument('-hf', '--hidden_feats', default=128, type=int,
                    help='The dimension of hidden tensor in the model')
# DD, DTD, DID, DSD, DCD,   DTTD, DTDTD, DIDID, DSDSD, DTITD, DITID
# 药物：DD、DC、DTD、DID、DSD、DTTD、DTDTD、DIDID、DSDSD、DTITD、DITID
# Drug, Protein, Disease, Side-effect
# meta_paths
DRUG_METAPATH_LIST = [
    #0
    ['drug_drug'],  # DD
    ['drug_chemical', 'chemical_drug'],  # DCD
    ['drug_protein', 'protein_drug'],  # DTD
    ['drug_disease', 'disease_drug'],  # DID
    ['drug_se', 'se_drug'],  # DSD

    # ['drug_protein', 'protein_protein', 'protein_drug'],  # DTTD
    # ['drug_protein', 'protein_drug', 'drug_protein', 'protein_drug'],  # DTDTD
    # ['drug_disease', 'disease_drug', 'drug_disease', 'disease_drug'],  # DIDID
    # ['drug_se', 'se_drug', 'drug_se', 'se_drug'],  # DSDSD
    # ['drug_protein', 'protein_disease', 'disease_protein', 'protein_drug'],  # DTITD
    # ['drug_disease', 'disease_protein', 'protein_disease', 'disease_drug'],  # DITID

    #1
    ['drug_drug', 'drug_protein', 'protein_drug'],  # DDTD
    ['drug_drug', 'drug_disease', 'disease_drug'],  # DDID
    ['drug_drug', 'drug_se', 'se_drug'],       # DDSD
    # ['drug_drug', 'drug_chemical', 'chemical_drug'],       # DDCD

    # ['drug_drug','drug_protein', 'protein_drug','drug_drug','drug_disease', 'disease_drug','drug_drug','drug_se', 'se_drug'],    # DDTD DDID DDSD



    #zhengMetapath
    # ['drug_drug'],  # DD
    # ['drug_protein', 'protein_drug'],  # DTD
    # ['drug_su', 'su_drug'],  # DSuD
    # ['drug_st', 'st_drug'],  # DStD
    # ['drug_s', 's_drug'],  # DSD
    # ['drug_protein', 'protein_protein', 'protein_drug'],  # DTTD

    # ['drug_drug', 'drug_protein', 'protein_drug'],  # DDTD
    # ['drug_drug', 'drug_su', 'su_drug'],  # DDDSuD
    # ['drug_drug', 'drug_st', 'st_drug'],  # DDDStD
    # ['drug_drug', 'drug_s', 's_drug'],  # DDDSD
]
# TT, TDT, TIT, TST,   TDDT, TITIT, TDTDT, TDIDT, TIDIT
# 靶标：TT、TS、TDT、TIT、TDDT、TTIT、TTDT、TITIT、TDTDT、TDIDT、TIDIT
TARGET_METAPATH_LIST = [

    ['protein_protein'],  # TT
    ['protein_sequence', 'sequence_protein'],  # TST
    ['protein_drug', 'drug_protein'],  # TDT
    ['protein_disease', 'disease_protein'],  # TIT


    # ['protein_drug', 'drug_drug', 'drug_protein'],  # TDDT
    # ['protein_drug', 'drug_protein', 'protein_drug', 'drug_protein'],  # TDTDT
    # ['protein_disease', 'disease_protein', 'protein_disease', 'disease_protein'],  # TITIT

    ['protein_drug', 'drug_disease', 'disease_drug', 'drug_protein'],  # TDIDT
    ['protein_disease', 'disease_drug', 'drug_disease', 'disease_protein'],  # TIDIT
    ['protein_protein', 'protein_drug', 'drug_protein'],  # TTDT
    ['protein_protein', 'protein_disease', 'disease_protein'],  # TTIT
    # ['protein_protein','protein_sequence', 'sequence_protein'],  # TTST

    # ['protein_protein','protein_drug', 'drug_protein', 'protein_protein','protein_disease', 'disease_protein','protein_protein','protein_sequence', 'sequence_protein'] # TTDT TTIT TTST


    #zhengMetapath
    # ['protein_protein'],  # TT
    # ['protein_drug', 'drug_protein'],  # TDT
    # ['protein_GO', 'GO_protein'],  # T(GO)T
    # ['protein_drug', 'drug_drug', 'drug_protein'],  # TDDT
    # ['protein_protein', 'protein_GO', 'GO_protein'],  # TT(GO)T
    # ['protein_protein', 'protein_drug', 'drug_protein'],  # TTDT

    # ['protein_drug', 'drug_su', 'su_drug', 'drug_protein'],  # TDSuDT
    # ['protein_drug', 'drug_st', 'st_drug', 'drug_protein'],  # TDStDT
    # ['protein_drug', 'drug_s', 's_drug', 'drug_protein'],  # TDSDT
    # ['protein_GO', 'GO_protein', 'protein_drug', 'drug_protein', 'protein_GO', 'GO_protein'],  # T(GO)TDT(GO)T
]
assert len(DRUG_METAPATH_LIST) == len(TARGET_METAPATH_LIST)

# model core params
HYPER_PARAM = dict(
    drug_metapath_list=DRUG_METAPATH_LIST,
    target_metapath_list=TARGET_METAPATH_LIST,
)
metapath_list = [HYPER_PARAM['drug_metapath_list'], HYPER_PARAM['target_metapath_list']]


def train():
    args = parser.parse_args()
    args.saved_path = args.saved_path + '_' + str(args.seed)
    # set_seed(args.seed)
    print(args)
    simplefilter(action='ignore', category=FutureWarning)
    if args.device_id:
        print('Training on GPU')
        device = th.device('cuda:{}'.format(args.device_id))
    else:
        print('Training on CPU')
        device = th.device('cpu')
    try:
        os.mkdir(args.saved_path)
    except:
        pass

    # df = pd.read_csv('./dataset/meidata/mat_drug_protein.csv', header=None).values
    df = pd.read_csv('./dataset/luodata/mat_drug_protein.csv', header=None).values
    # df = pd.read_csv('./dataset/zhengdata/mat_drug_target.csv', header=None).values
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    data = data.astype('int64')

    # 所有正样本
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    print('all positive sample number:', len(data_pos))
    # 所有负样本
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    print('all negative sample number:', len(data_neg))
    assert len(data) == len(data_pos) + len(data_neg)

    fold = 1
    # 构造拓扑图模块 pos:8750 neg:4348402
    whole_negative_index = data_neg
    # 从whole_negative_index中随机选择len(data_pos)个负样本
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(data_pos),
                                             replace=False)
    print('select positive sample number:', len(negative_sample_index))

    data_set = np.zeros((len(negative_sample_index) + len(data_pos), 3),
                        dtype=int)
    count = 0
    for i in range(len(data_pos)):
        data_set[count][0] = data_pos[i][0]
        data_set[count][1] = data_pos[i][1]
        data_set[count][2] = 1
        count += 1

    # 得到dti_cledge.txt
    f = open("dti_cledge.txt", "w", encoding="utf-8")
    for i in range(count):
        for j in range(count):
            if data_set[i][0] == data_set[j][0] or data_set[i][1] == data_set[j][1]:
                f.write(f"{i}\t{j}\n")

    data_negative_sample_index = np.zeros((len(negative_sample_index), 3), dtype=int)
    countNeg = 0
    for i in range(len(negative_sample_index)):
        data_set[count][0] = whole_negative_index[negative_sample_index[i]][0]
        data_set[count][1] = whole_negative_index[negative_sample_index[i]][1]
        data_set[count][2] = 0
        data_negative_sample_index[countNeg][0] = whole_negative_index[negative_sample_index[i]][0]
        data_negative_sample_index[countNeg][1] = whole_negative_index[negative_sample_index[i]][1]
        data_negative_sample_index[countNeg][2] = 0
        count += 1
        countNeg += 1

    # 得到dti_index.txt
    f = open(f"dti_index.txt", "w", encoding="utf-8")
    for i in data_set:
        f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

    dateset = data_set

    # 得到dtiedge.txt
    f = open("dtiedge.txt", "w", encoding="utf-8")
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):
            if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
                f.write(f"{i}\t{j}\n")
    f.close()

    # load clGraph
    cl = get_clGraph(dateset, "dti").to(device)

    # true label
    label = th.tensor(dateset[:, 2:3])
    label = label.type(th.LongTensor)
    label = label.to(device)

    set1 = []
    set2 = []
    # finished 5-CV, four for training, one for testing
    skf = StratifiedKFold(n_splits=args.nfold, shuffle=True)
    for train_index, test_index in skf.split(dateset[:, :2], dateset[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    # 5-CV, begin
    pred_result = np.zeros(df.shape)
    label_true = th.tensor(df).float().to(device)
    all_auc = []
    all_aupr = []
    all_acc = []
    all_f1 = []
    all_pre = []
    all_rec = []
    all_spe = []
    print_reuslt = []
    for i in range(len(set1)):
        best_AUC = 0
        best_AUPR = 0
        best_pred = []
        mask_train = set1[i]
        mask_test = set2[i]
        print('{}-Cross Validation: Fold {}'.format(args.nfold, fold))
        # load hetero_graph
        g = loadluodata()
        # remove test from test_set
        test_id = th.tensor(dateset[mask_test], dtype=torch.int64)
        test_id = test_id.cpu().numpy()
        test_pos_id = []
        for j in range(test_id.shape[0]):
            if test_id[j][2] == 1:
                test_pos_id.append([test_id[j][0], test_id[j][1]])
        test_pos_id = np.array(test_pos_id)
        g = remove_graph(g, test_pos_id).to(device)
        # extracted drug and disease features
        feature = {'drug': g.nodes['drug'].data['h'], 'protein': g.nodes['protein'].data['h']}

        # init model
        model = MMDACL_bio(in_dim=feature['drug'].shape[1], hidden_dim=args.hidden_feats,
                         num_metapaths=len(metapath_list[0]))
        model.to(device)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
        optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer,
                                                         base_lr=0.1 * args.learning_rate,
                                                         max_lr=args.learning_rate,
                                                         gamma=0.995,
                                                         step_size_up=20,
                                                         mode="exp_range",
                                                         cycle_momentum=False)

        stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)
        for epoch in range(1, args.epoch + 1):
            model.train()
            score, core_loss, attn_out_drug, attn_out_protein= model(g, feature, metapath_list, cl, mask_train, dateset, label[mask_train])
            reg = get_L2reg(model.parameters())
            loss = core_loss + 0.001 * reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optim_scheduler.step()
            model.eval()
            AUC_, AUPR_ = get_metrics_auc(label[mask_train].cpu().detach().numpy(),
                                          score[:, 1:].cpu().detach().numpy())
            # AUC_, AUPR_, acc_tr, f1_tr, pre_tr, rec_tr, spe_tr = get_metrics(label[mask_train].cpu().detach().numpy().flatten(), score[:, 1:].cpu().detach().numpy().flatten())
            early_stop = stopper.step(loss.item(), AUC_, model)
            pred_test = model(g, feature, metapath_list, cl, mask_test, dateset, label[mask_test], iftrain=False,
                               attn_out_drug=attn_out_drug, attn_out_protein=attn_out_protein)
            AUC__, AUPR__ = get_metrics_auc(label[mask_test].cpu().detach().numpy(),
                                            pred_test[:, 1:].cpu().detach().numpy())
            # AUC__, AUPR__, acc_te, f1_te, pre_te, rec_te, spe_te = get_metrics(label[mask_test].cpu().detach().numpy().flatten(), pred_test[:, 1:].cpu().detach().numpy().flatten())
            if AUC__ > best_AUC:
                best_AUC = AUC__
                best_AUPR = AUPR__
                best_pred = pred_test
            if epoch % 50 == 0:
                print(
                    'Epoch {} Loss: {:.4f}; Train AUC: {:.4f}; Train AUPR: {:.4f}; Test AUC: {:.4f}; Test AUPR: {:.4f};'.format(
                        epoch,
                        loss.item(),
                        AUC_,
                        AUPR_,
                        best_AUC,
                        best_AUPR,
                        ))
                print('-' * 50)
                if early_stop:
                    break

        stopper.load_checkpoint(model)
        model.eval()
        # AUC, AUPR = get_metrics_auc(label[mask_test].cpu().detach().numpy(),
        #                             pred_test[:, 1:].cpu().detach().numpy())
        AUC, AUPR, acc, f1, pre, rec, spe = get_metrics(label[mask_test].cpu().detach().numpy().flatten(), best_pred[:, 1:].cpu().detach().numpy().flatten())
        all_auc.append(AUC)
        all_aupr.append(AUPR)
        all_acc.append(acc)
        all_f1.append(f1)
        all_pre.append(pre)
        all_rec.append(rec)
        all_spe.append(spe)
        for j in range(test_id.shape[0]):
            pred_result[test_id[j][0], test_id[j][1]] = best_pred[:, 1:][j]
        fold += 1
    mean_auc = sum(all_auc) / len(all_auc)
    mean_aupr = sum(all_aupr) / len(all_aupr)
    mean_acc = sum(all_acc) / len(all_acc)
    mean_f1 = sum(all_f1) / len(all_f1)
    mean_pre = sum(all_pre) / len(all_pre)
    mean_rec = sum(all_rec) / len(all_rec)
    mean_spe = sum(all_spe) / len(all_spe)
    print(
        'Overall: AUC: {:.4f}; AUPR: {:.4f}; F1: {:.4f}; Acc: {:.4f}; Recall {:.4f}; Specificity {:.4f}; Precision {:.4f}'.
            format(mean_auc, mean_aupr, mean_f1, mean_acc, mean_rec, mean_spe, mean_pre))

    print_reuslt.append([mean_auc, mean_aupr, mean_f1, mean_acc, mean_rec, mean_spe, mean_pre])
    print_reuslt = np.array(print_reuslt)
    np.savetxt('print_reuslt.csv', print_reuslt, delimiter=',')

    return mean_auc, mean_aupr, mean_f1, mean_acc, mean_rec, mean_spe, mean_pre, pred_result

if __name__ == '__main__':
    all_reuslt = []
    for i in range(5):
        mean_auc, mean_aupr, mean_f1, mean_acc, mean_rec, mean_spe, mean_pre, pred_result = train()
        np.savetxt('pred_result' + str(i) + '.csv', pred_result, delimiter=',')
        all_reuslt.append([mean_auc, mean_aupr, mean_f1, mean_acc, mean_rec, mean_spe, mean_pre])
    all_reuslt = np.array(all_reuslt)
    np.savetxt('all_result.csv', all_reuslt, delimiter=',')
