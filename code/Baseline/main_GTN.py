import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.baselines import GTN
import pdb
import pandas as pd
import pickle
import argparse
from model.modules import LinkPrediction_minibatch, LinkPrediction_fullbatch
from experiment.link_prediction import link_prediction_minibatch, link_prediction_fullbatch
from utils import metapath2str, get_metapath_g, get_khop_g, load_data_nc, load_data_lp, \
    get_save_path, load_base_config, load_model_config

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='react',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--repeat', type=int, default=1,
                        help='repeat the training and testing for N times')
    parser.add_argument('--model', type=str, default='GTN', help='name of model')

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    dir_path_list = []
    for _ in range(args.repeat):
        dir_path_list.append(get_save_path(args))

    test_auroc_list = []
    test_ap_list = []
    test_ar_list=[]

    with open('data/' + args.dataset + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    num_nodes = edges[0].shape[0]

    users = pd.read_csv('data/react/index/user_index.txt', sep='\t', header=None, names=['user_id', 'user'],
                        keep_default_na=False, encoding='utf-8')
    num_users=len(users)
    features_0 =  np.random.rand(num_users, 10)
    features_1 = np.load('data/react/features_1.npy')
    features_2 = np.load('data/react/features_2.npy')
    features_3 = np.load('data/react/features_3.npy')
    features_list= [features_0, features_1, features_2, features_3]

    for i, edge in enumerate(edges[0]):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    in_dims = [features.shape[1] for features in features_list]
    node_features = torch.from_numpy(in_dims).type(torch.FloatTensor)

    (g_train, g_val, g_test), in_dim_dict, (train_eid_dict, val_eid_dict, test_eid_dict), (
        val_neg_uv, test_neg_uv) = load_data_lp()
    train_eid_dict = {metapath2str([g_train.to_canonical_etype(k)]): v for k, v in train_eid_dict.items()}
    val_eid_dict = {metapath2str([g_val.to_canonical_etype(k)]): v for k, v in val_eid_dict.items()}
    test_eid_dict = {metapath2str([g_test.to_canonical_etype(k)]): v for k, v in test_eid_dict.items()}
    target_etype = list(train_eid_dict.keys())[0]

    final_f1 = 0
    for l in range(1):

        model = GTN.GTN_lp(num_edge=A.shape[-1],
                    num_channels=num_channels,
                    w_in=node_features.shape[1],
                    w_out=node_dim,
                    num_layers=num_layers,
                    norm=norm)

        model_lp = LinkPrediction_minibatch(model, args.hidden_dim, target_etype)

        minibatch_flag = True

        if minibatch_flag:
            test_auroc, test_ap,test_ar = link_prediction_minibatch(model_lp, g_train, g_val, g_test, train_eid_dict,
                                                            val_eid_dict, test_eid_dict, val_neg_uv, test_neg_uv,
                                                            dir_path_list[i], args)
        else:
            test_auroc, test_ap,test_ar = link_prediction_fullbatch(model_lp, g_train, g_val, g_test, train_eid_dict,
                                                            val_eid_dict, test_eid_dict, val_neg_uv, test_neg_uv,
                                                            dir_path_list[i], args)

        test_auroc_list.append(test_auroc)
        test_ap_list.append(test_ap)
        test_ar_list.append(test_ar)

        print("--------------------------------")
        if args.repeat > 1:
            print("ROC-AUC_MEAN\tROC-AUC_STDEV\tPR-AUC_MEAN\tPR-AUC_STDEV")
            print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(np.mean(test_auroc_list), np.std(test_auroc_list, ddof=0),
                                                          np.mean(test_ap_list), np.std(test_ap_list, ddof=0),
                                                          np.mean(test_ar_list), np.std(test_ar_list, ddof=0))
                  )
        else:
            print("args.repeat <= 1, not calculating the average and the standard deviation of scores")