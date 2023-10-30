import time
import argparse
import pandas as pd
import torch,json
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from utils.pytorchtools import EarlyStopping
from utils.data import load_LastFM_data
from utils.tools import index_generator, parse_minibatch_LastFM
from magnn_model import MAGNN_lp

# Params
num_ntype = 4
dropout_rate = 0.5
lr = 0.001
weight_decay = 0.001
etypes_lists = [[[0, 0], [1,1],[6],[1,2,2,1],[0, 2, 2, 0],[1,3,1]],
                [[4, 4], [5,5], [5, 2, 2, 5],[4, 2, 2, 4],[5,3, 5],[7]]]


use_masks = [[True, False, True, True, False, True],
             [True, False, True, True, False, True]]

no_masks = [[False] * 6, [False] * 6]

basic_repo = './facebook_react'
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

def run_model_LastFM(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    adjlists_ip, edge_metapath_indices_list_ip,  features_list, adjM, type_mask, train_val_test_pos_issue_pr, train_val_test_neg_issue_pr = load_LastFM_data()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    in_dims = []
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))
    elif feats_type == 2:
        for i in range(num_ntype):
            num_nodes = (type_mask == i).sum()
            features_list.append(torch.zeros((num_nodes, features_list[i].shape[1])).to(device))
    train_pos_issue_pr = train_val_test_pos_issue_pr['train_pos_issue_pr']
    val_pos_issue_pr = train_val_test_pos_issue_pr['val_pos_issue_pr']
    test_pos_issue_pr = train_val_test_pos_issue_pr['test_pos_issue_pr']
    train_neg_issue_pr = train_val_test_neg_issue_pr['train_neg_issue_pr']
    val_neg_issue_pr = train_val_test_neg_issue_pr['val_neg_issue_pr']
    test_neg_issue_pr = train_val_test_neg_issue_pr['test_neg_issue_pr']

    y_true_test = np.array([1] * len(train_pos_issue_pr) + [0] * len(train_neg_issue_pr))
    auc_list = []
    ap_list = []
    ar_list = []
    af1_list = []
    for _ in range(repeat):
        net = MAGNN_lp([6,6 ], 8, etypes_lists, in_dims, hidden_dim, hidden_dim,num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_issue_pr))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_issue_pr), shuffle=False)
        val_neg_generator = index_generator(batch_size=batch_size, num_data=len(val_neg_issue_pr), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):

                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_issue_pr_batch = train_pos_issue_pr[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_issue_pr), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_issue_pr_batch = train_neg_issue_pr[train_neg_idx_batch].tolist()

                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ip, edge_metapath_indices_list_ip, train_pos_issue_pr_batch, device, neighbor_samples, use_masks, num_issues)
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ip, edge_metapath_indices_list_ip, train_neg_issue_pr_batch, device, neighbor_samples, no_masks, num_issues)

                t1 = time.time()
                dur1.append(t1 - t0)
                [pos_embedding_user, pos_embedding_artist], _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_pos_idx_batch_mapped_lists))
                [neg_embedding_user, neg_embedding_artist], _ = net(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist)
                pos_out = pos_out.squeeze(dim=-1).float()
                neg_out = neg_out.squeeze(dim=-1).float()
                # train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
                pos_target=np.ones(shape=pos_out.shape)
                pos_target=pos_target.astype(np.float32)
                neg_target=np.zeros(shape=neg_out.shape)
                neg_target = neg_target.astype(np.float32)
                pos_target = torch.tensor(pos_target).to(device)
                neg_target = torch.tensor(neg_target).to(device)

                train_loss = F.binary_cross_entropy(torch.sigmoid(pos_out), pos_target)+F.binary_cross_entropy(torch.sigmoid(neg_out), neg_target)


                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    # val_neg_batch = val_neg_generator.next()
                    val_pos_issue_pr_batch = val_pos_issue_pr[val_idx_batch].tolist()
                    val_neg_issue_pr_batch = val_neg_issue_pr[val_idx_batch].tolist()
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                        adjlists_ip, edge_metapath_indices_list_ip, val_pos_issue_pr_batch, device, neighbor_samples, no_masks, num_issues)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                        adjlists_ip, edge_metapath_indices_list_ip, val_neg_issue_pr_batch, device, neighbor_samples, no_masks, num_issues)

                    [pos_embedding_user, pos_embedding_artist], _ = net(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                    [neg_embedding_user, neg_embedding_artist], _ = net(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                    pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                    pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                    neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                    neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist)
                    pos_target = np.ones(shape=pos_out.shape)
                    pos_target = pos_target.astype(np.float32)
                    neg_target = np.zeros(shape=neg_out.shape)
                    neg_target = neg_target.astype(np.float32)
                    pos_target = torch.tensor(pos_target).to(device)
                    neg_target = torch.tensor(neg_target).to(device)
                    val_loss.append(F.binary_cross_entropy(F.sigmoid(pos_out), pos_target)+F.binary_cross_entropy(F.sigmoid(neg_out), neg_target))
                val_loss = torch.mean(torch.tensor(val_loss))

            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_issue_pr), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_issue_pr_batch = test_pos_issue_pr[test_idx_batch].tolist()
                test_neg_issue_pr_batch = test_neg_issue_pr[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ip, edge_metapath_indices_list_ip, test_pos_issue_pr_batch, device, neighbor_samples, no_masks, num_issues)
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ip, edge_metapath_indices_list_ip, test_neg_issue_pr_batch, device, neighbor_samples, no_masks, num_issues)

                [pos_embedding_user, pos_embedding_artist], _ = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
                [neg_embedding_user, neg_embedding_artist], _ = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist).flatten()
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))

            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()

        y_proba_test=np.where(y_proba_test>=0.5,1,0)
        auc= accuracy_score(y_true_test, y_proba_test.astype(int))
        precision= precision_score(y_true_test, y_proba_test)
        recall= recall_score(y_true_test, y_proba_test)

        auc_list.append(auc)
        ap_list.append(precision)
        ar_list.append(recall)
        af1_list.append(2*precision*recall/(precision+recall))

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}'.format(np.mean(auc_list)))
    print('AP_mean = {}'.format(np.mean(ap_list)))
    print('AR_mean = {}'.format(np.mean(ar_list)))
    print('AF1_mean = {},'.format(np.mean(af1_list)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=16, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='checkpoint', help='Postfix for the saved model and result. Default is LastFM.')

    args = ap.parse_args()
    run_model_LastFM(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)