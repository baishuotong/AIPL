import pathlib,json
import numpy as np
import scipy.sparse
import scipy.io,os
import pandas as pd
import pickle
import networkx as nx
from sklearn.model_selection import train_test_split
'''
构建邻接矩阵
'''

basic_repo='./facebook_react/'
save_path='./facebook_react/processed/'
basic_path='./work3/facebook_react/'

######### 构建矩阵
user_repo=pd.read_csv(basic_repo+'/edge_index/user_repo_index.txt', sep='\t', header=None, names=['user_id', 'repo_id'], keep_default_na=False, encoding='utf-8')
user_issue=pd.read_csv(basic_repo+'/edge_index/user_issue_index.txt', sep='\t', header=None, names=['user_id', 'issue_id'], keep_default_na=False, encoding='utf-8')
user_pr=pd.read_csv(basic_repo+'/edge_index/user_pr_index.txt', sep='\t', header=None, names=['user_id', 'pr_id'], keep_default_na=False, encoding='utf-8')

repo_repo = pd.read_csv(basic_repo+'/edge_index/repo_repo_index.txt', sep='\t', header=None, names=['repo_id', 'repo_id_1'], keep_default_na=False, encoding='utf-8')
repo_issue = pd.read_csv(basic_repo+'/edge_index/repo_issue_index.txt', sep='\t', header=None, names=['repo_id', 'issue_id'], keep_default_na=False, encoding='utf-8')
repo_pr = pd.read_csv(basic_repo+'/edge_index/repo_pr_index.txt', sep='\t', header=None, names=['repo_id', 'pr_id'], keep_default_na=False, encoding='utf-8')

issue_issue=pd.read_csv(basic_repo+'/edge_index/issue_issue_index.txt', sep='\t', header=None, names=['issue_id', 'issue_id_1'], keep_default_na=False, encoding='utf-8')
issue_pr=pd.read_csv(basic_repo+'/edge_index/issue_pr_index.txt', sep='\t', header=None, names=['issue_id', 'pr_id'], keep_default_na=False, encoding='utf-8')
pr_pr=pd.read_csv(basic_repo+'/edge_index/pr_pr_index.txt', sep='\t', header=None, names=['pr_id', 'pr_id_1'], keep_default_na=False, encoding='utf-8')

issues = pd.read_csv(basic_repo+'/index/issue_index.txt', sep='\t', header=None, names=['issue_id', 'issue'], keep_default_na=False, encoding='utf-8')
prs = pd.read_csv(basic_repo+'/index/pr_index.txt', sep='\t', header=None, names=['pr_id', 'pr'], keep_default_na=False, encoding='utf-8')
users = pd.read_csv(basic_repo+'/index/user_index.txt', sep='\t', header=None, names=['user_id', 'user'], keep_default_na=False, encoding='utf-8')
repos= pd.read_csv(basic_repo+'/index/repo_index.txt', sep='\t', header=None, names=['repo_id', 'repo'], keep_default_na=False, encoding='utf-8')

num_users=len(users)
num_repos=len(repos)
num_issues=len(issues)
num_prs=len(prs)

dim = len(users) + len(repos) + len(issues) + len(prs)

########### 保存每个节点的type all nodes (users, repos, issues and prs) type labels
type_mask = np.zeros((dim), dtype=int)
type_mask[len(users):len(users)+len(repos)] = 1
type_mask[len(users)+len(repos):len(users)+len(repos)+len(issues)] = 2
type_mask[len(users)+len(repos)+len(issues):] = 3

np.save(save_path + 'node_types.npy', type_mask)

#将每一个节点对应 的标号映射到列表的位置上
user_id_mapping = {row['user_id']: i for i, row in users.iterrows()}
repo_id_mapping = {row['repo_id']: i + len(users) for i, row in repos.iterrows()}
issue_id_mapping = {row['issue_id']: i + len(users)+len(repos) for i, row in issues.iterrows()}
pr_id_mapping = {row['pr_id']: i + len(users)+len(repos)+len(issues) for i, row in prs.iterrows()}

####### 构建邻接矩阵

adjM = np.zeros((dim, dim), dtype=int)

for _, row in user_repo.iterrows():
    idx1 = user_id_mapping[row['user_id']]
    idx2 = repo_id_mapping[row['repo_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in user_issue.iterrows():
    idx1 = user_id_mapping[row['user_id']]
    idx2 = issue_id_mapping[row['issue_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in user_pr.iterrows():
    idx1 = user_id_mapping[row['user_id']]
    idx2 = pr_id_mapping[row['pr_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

for _, row in repo_repo.iterrows():
    idx1 = repo_id_mapping[row['repo_id']]
    idx2 = repo_id_mapping[row['repo_id_1']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in repo_issue.iterrows():
    idx1 = repo_id_mapping[row['repo_id']]
    idx2 = issue_id_mapping[row['issue_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in repo_pr.iterrows():
    idx1 = repo_id_mapping[row['repo_id']]
    idx2 = pr_id_mapping[row['pr_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

for _, row in issue_issue.iterrows():
    idx1 = issue_id_mapping[row['issue_id']]
    idx2 = issue_id_mapping[row['issue_id_1']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in issue_pr.iterrows():
    idx1 = issue_id_mapping[row['issue_id']]
    idx2 = pr_id_mapping[row['pr_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in pr_pr.iterrows():
    idx1 = pr_id_mapping[row['pr_id']]
    idx2 = pr_id_mapping[row['pr_id_1']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

##### 保存邻接矩阵
scipy.sparse.save_npz(save_path + 'adjM.npz', scipy.sparse.csr_matrix(adjM))

## 读取邻接矩阵
adjM = scipy.sparse.load_npz(save_path + 'adjM.npz')
adjM=adjM.toarray()

'''
构建特征
'''

### 生成每个节点的特征，title和项目描述 （，50)
# #1) repo
features_repo = np.zeros((len(repos), 50))
repo_info = pd.read_csv(basic_repo + '/repo_information.csv', encoding='utf-8')

for i, row in repos.iterrows():
    repo_vector = repo_info['repo_vector'].loc[i].replace('[', '').replace(']', '').replace('\n', '')
    features_repo[i] = np.array([float(x) for x in repo_vector.split(' ') if x != ''])

# #2）issue
features_issue = np.zeros((len(issues), 50))

i=0
for dir_path in os.listdir(basic_path):

    df_i = pd.read_csv(basic_path + dir_path + '/issue_vector.csv')

    for index in range(len(df_i)):
        issue_vector = df_i['title_vector'].loc[index].replace('[', '').replace(']', '').replace('\n', '')
        features_issue[i] = np.array([float(x) for x in issue_vector.split(' ') if x != ''])
        i+=1

# #3) pr
features_pr = np.zeros((len(prs), 50))

i=0
for dir_path in os.listdir(basic_path):

    df_p = pd.read_csv(basic_path + dir_path + '/pr_vector.csv')
    for index in range(len(df_p)):
        pr_vector = df_p['title_vector'].loc[index].replace('[', '').replace(']', '').replace('\n', '')
        features_pr[i] = np.array([float(x) for x in pr_vector.split(' ') if x != ''])
        i += 1


###### 保存所有节点的特征 all nodes (users, repos, issues and prs) features

np.save(save_path + 'features_{}.npy'.format(1), features_repo)
np.save(save_path + 'features_{}.npy'.format(2), features_issue)
np.save(save_path + 'features_{}.npy'.format(3), features_pr)


