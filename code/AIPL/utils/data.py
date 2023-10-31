import networkx as nx
import numpy as np
import scipy.sparse
import pickle
import pandas as pd

def load_LastFM_data(prefix='./facebook_react\processed'):
    basic_repo='./facebook_react'
    issues = pd.read_csv(basic_repo + '/index/issue_index.txt', sep='\t', header=None, names=['issue_id', 'issue'],
                         keep_default_na=False, encoding='utf-8')
    prs = pd.read_csv(basic_repo + '/index/pr_index.txt', sep='\t', header=None, names=['pr_id', 'pr'],
                      keep_default_na=False, encoding='utf-8')
    users = pd.read_csv(basic_repo + '/index/user_index.txt', sep='\t', header=None, names=['user_id', 'user'],
                        keep_default_na=False, encoding='utf-8')
    repos = pd.read_csv(basic_repo + '/index/repo_index.txt', sep='\t', header=None, names=['repo_id', 'repo'],
                        keep_default_na=False, encoding='utf-8')

    num_users = len(users)
    num_repos = len(repos)
    num_issues = len(issues)
    num_prs = len(prs)

    in_file = open(prefix + '/2/2-0-2.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/2/2-1-2.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    # in_file = open(prefix + '/2/2-2.adjlist', 'r')
    # adjlist02 = [line.strip() for line in in_file]
    # adjlist02 = adjlist02
    # in_file.close()
    in_file = open(prefix + '/2/2-1-0-1-2.adjlist', 'r')
    adjlist03 = [line.strip() for line in in_file]
    adjlist03 = adjlist03
    in_file.close()
    # in_file = open(prefix + '/2/2-0-1-0-2.adjlist', 'r')
    # adjlis04 = [line.strip() for line in in_file]
    # adjlist04 = adjlis04
    # in_file.close()
    in_file = open(prefix + '/2/2-1-1-2.adjlist', 'r')
    adjlist05 = [line.strip() for line in in_file]
    adjlist05 = adjlist05
    in_file.close()

    in_file = open(prefix + '/3/3-0-3.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/3/3-1-3.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/3/3-1-0-1-3.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()
    # in_file = open(prefix + '/3/3-0-1-0-3.adjlist', 'r')
    # adjlist13 = [line.strip() for line in in_file]
    # adjlist13 = adjlist13
    # in_file.close()
    in_file = open(prefix + '/3/3-1-1-3.adjlist', 'r')
    adjlist14 = [line.strip() for line in in_file]
    adjlist14 = adjlist14
    in_file.close()
    # in_file = open(prefix + '/3/3-3.adjlist', 'r')
    # adjlist15 = [line.strip() for line in in_file]
    # adjlist15 = adjlist15
    # in_file.close()


    in_file = open(prefix + '/2/2-0-2_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/2/2-1-2_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + '/2/2-2_idx.pickle', 'rb')
    # idx02 = pickle.load(in_file)
    # in_file.close()
    in_file = open(prefix + '/2/2-1-0-1-2_idx.pickle', 'rb')
    idx03 = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + '/2/2-0-1-0-2_idx.pickle', 'rb')
    # idx04 = pickle.load(in_file)
    # in_file.close()
    in_file = open(prefix + '/2/2-1-1-2_idx.pickle', 'rb')
    idx05 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-3_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/3/3-1-3_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/3/3-1-0-1-3_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + '/3/3-0-1-0-3_idx.pickle', 'rb')
    # idx13 = pickle.load(in_file)
    # in_file.close()
    in_file = open(prefix + '/3/3-1-1-3_idx.pickle', 'rb')
    idx14 = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + '/3/3-3_idx.pickle', 'rb')
    # idx15 = pickle.load(in_file)
    # in_file.close()

    ##### 传入节点特征
    features_0 =  np.random.rand(num_users, 10)
    features_1 = np.load(prefix + '/features_1.npy')
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.load(prefix + '/features_3.npy')

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_issue_pr = np.load(prefix + '/train_val_test_idx.npz')
    train_val_test_neg_issue_pr = np.load(prefix + '/train_val_test_neg_issue_pr.npz')


    # return [[adjlist00, adjlist01, adjlist02, adjlist03, adjlist04,adjlist05],[adjlist10, adjlist11, adjlist12,adjlist13,adjlist14,adjlist15]],\
    #        [[idx00, idx01, idx02, idx03,idx04,idx05], [idx10, idx11, idx12,idx13,idx14,idx15]],\
    # [features_0, features_1, features_2, features_3],\
    #        adjM, type_mask, train_val_test_pos_issue_pr, train_val_test_neg_issue_pr
    return [[adjlist00, adjlist01, adjlist03, adjlist05],[adjlist10, adjlist11, adjlist12,adjlist14]],\
           [[idx00, idx01, idx03,idx05], [idx10, idx11, idx12,idx14]],\
    [features_0, features_1, features_2, features_3],\
           adjM, type_mask, train_val_test_pos_issue_pr, train_val_test_neg_issue_pr

