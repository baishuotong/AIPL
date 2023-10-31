import numpy as np
import pandas as pd
import pickle,pathlib,scipy.sparse

basic_repo='./facebook_react'
save_path='./facebook_react/processed/'
basic_path='./work3/facebook_react/'

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

###### 读取矩阵
adjM = scipy.sparse.load_npz(save_path + 'adjM.npz')### 读取
adjM=adjM.toarray()

user_repo_list={i: adjM[i, len(users):len(users)+len(repos)].nonzero()[0] for i in range(len(users))}
user_issue_list={i:adjM[i, len(users)+len(repos):len(users)+len(repos)+len(issues)].nonzero()[0] for i in range(len(users))}
user_pr_list={i:adjM[i, len(users)+len(repos)+len(issues):].nonzero()[0] for i in range(len(users))}
repo_repo_list= {i: adjM[len(users)+i,len(users):len(users)+len(repos)].nonzero()[0] for i in range(len(repos))}
repo_user_list= {i: adjM[len(users)+i,:len(users)].nonzero()[0] for i in range(len(repos))}
repo_issue_list={i:adjM[len(users)+i,len(users)+len(repos):len(users)+len(repos)+len(issues)].nonzero()[0] for i in range(len(repos))}
repo_pr_list={i:adjM[len(users)+i,len(users)+len(repos)+len(issues):].nonzero()[0] for i in range(len(repos))}
issue_issue_list={i:adjM[len(users)+len(repos)+i,len(users)+len(repos):len(users)+len(repos)+len(issues)].nonzero()[0] for i in range(len(issues))}
issue_user_list={i:adjM[len(users)+len(repos)+i, :len(users)].nonzero()[0]for i in range(len(issues))}
issue_repo_list={i:adjM[len(users)+len(repos)+i, len(users):len(users)+len(repos)].nonzero()[0]for i in range(len(issues))}
issue_pr_list={i:adjM[len(users)+len(repos)+i,len(users)+len(repos)+len(issues):].nonzero()[0] for i in range(len(issues))}
pr_pr_list={i:adjM[len(users)+len(repos)+len(issues)+i,len(users)+len(repos)+len(issues):].nonzero()[0] for i in range(len(prs))}
pr_user_list={i:adjM[len(users)+len(repos)+len(issues)+i,:len(users)].nonzero()[0] for i in range(len(prs))}
pr_repo_list={i:adjM[len(users)+len(repos)+len(issues)+i,len(users):len(users)+len(repos)].nonzero()[0] for i in range(len(prs))}
pr_issue_list={i:adjM[len(users)+len(repos)+len(issues)+i,len(users)+len(repos):len(users)+len(repos)+len(issues)].nonzero()[0] for i in range(len(prs))}

'''
开始创建
'''


## 2-2 issue-issue
i_i = issue_issue.to_numpy(dtype=np.int32)
i_i=np.array(i_i)
i_i += len(users)+len(repos)
sorted_index = sorted(list(range(len(i_i))), key=lambda i : i_i[i].tolist())
i_i = i_i[sorted_index]

##3-3 pr-pr
p_p = pr_pr.to_numpy(dtype=np.int32)
p_p += len(users)+len(repos)+len(issues)
sorted_index = sorted(list(range(len(p_p))), key=lambda i : p_p[i].tolist())
p_p = p_p[sorted_index]

# 2-0-2 i_u_i
i_u_i = []
for u, i_list in user_issue_list.items():
    i_u_i.extend([(i1, u, i2) for i1 in i_list for i2 in i_list])
i_u_i = np.array(i_u_i)
i_u_i[:, [0, 2]] += len(users)+len(repos) ### 中间的index从项目数量开始计算
sorted_index = sorted(list(range(len(i_u_i))), key=lambda i :i_u_i[i, [0, 2, 1]].tolist())
i_u_i = i_u_i[sorted_index]

# 3-0-3 p_u_p
p_u_p = []
for u, p_list in user_pr_list.items():
    p_u_p.extend([(p1, u, p2) for p1 in p_list for p2 in p_list])
p_u_p = np.array(p_u_p)
p_u_p[:, [0, 2]] += len(users)+len(repos)+len(issues) ### 中间的index从项目数量开始计算
sorted_index = sorted(list(range(len(p_u_p))), key=lambda i :p_u_p[i, [0, 2, 1]].tolist())
p_u_p = p_u_p[sorted_index]

# 2-1-2 iss_repo_iss
i_r_i = []
for r, i_list in repo_issue_list.items():
    i_r_i.extend([(i1, r, i2) for i1 in i_list for i2 in i_list])
i_r_i = np.array(i_r_i) # (6713241, 3)
i_r_i += len(users)
i_r_i[:, [0, 2]] += len(repos) ### 中间的index从项目数量开始计算
sorted_index = sorted(list(range(len(i_r_i))), key=lambda i :i_r_i[i, [0, 2, 1]].tolist())
i_r_i = i_r_i[sorted_index]

# 3-1-3 p_repo_p
p_r_p = []
for r, p_list in repo_pr_list.items():
    p_r_p.extend([(p1, r, p2) for p1 in p_list for p2 in p_list])
p_r_p = np.array(p_r_p) # (867560, 3)
p_r_p += len(users)
p_r_p[:, [0, 2]] += len(repos)+len(issues) ### 中间的index从项目数量开始计算
sorted_index = sorted(list(range(len(p_r_p))), key=lambda i :p_r_p[i, [0, 2, 1]].tolist())
p_r_p = p_r_p[sorted_index]

#2-1-1-2 isssue_repo_repo_issue

i_r_r_i = []
for i, r_list in issue_repo_list.items():
    for r1 in r_list:
        for r2 in repo_repo_list[r1]:
            for i2 in repo_issue_list[r2]:
                i_r_r_i.extend([(i, r1, r2, i2)])
i_r_r_i = np.array(i_r_r_i) # (3325, 3)
i_r_r_i[:, [0, 3]]+= len(repos)+len(users) ### 中间的index从项目数量开始计算
i_r_r_i[:,[1, 2]]+= len(users)
sorted_index = sorted(list(range(len(i_r_r_i))), key=lambda i :i_r_r_i[i, [0, 2, 1, 3]].tolist())
i_r_r_i = i_r_r_i[sorted_index]

#3-1-1-3 pr_repo_repo_pr
p_r_r_p = []
for p, r_list in issue_repo_list.items():
    for r1 in r_list:
        for r2 in repo_repo_list[r1]:
            for p2 in repo_pr_list[r2]:
                p_r_r_p.extend([(p, r1, r2, p2)])
p_r_r_p = np.array(p_r_r_p) # (3325, 3)
p_r_r_p[:, [0, 3]]+= len(repos)+len(users) ### 中间的index从项目数量开始计算
p_r_r_p[:,[1, 2]]+= len(users)
sorted_index = sorted(list(range(len(p_r_r_p))), key=lambda i :p_r_r_p[i, [0, 2, 1, 3]].tolist())
p_r_r_p = p_r_r_p[sorted_index]

# 2-0-1-0-2 i_u_r_u_i

u_r_u = []
for r, u_list in repo_user_list.items():
    for u1 in u_list:
            u_r_u.extend([(u1, r, u2) for u2 in u_list])

u_r_u = np.array(u_r_u)
u_r_u[:, 1] += len(users)
sorted_index = sorted(list(range(len(u_r_u))), key=lambda i :u_r_u[i, [0, 2, 1]].tolist())
u_r_u = u_r_u[sorted_index]

i_u_r_u_i = []

for u1, r, u2 in u_r_u[0:]:

    if len(user_issue_list[int(u1)]) == 0 or len(user_issue_list[int(u2)]) == 0:
        continue

    candidate_i1_list = np.random.choice(len(user_issue_list[int(u1)]), int(len(user_issue_list[int(u1)])), replace=False)
    candidate_i2_list = np.random.choice(len(user_issue_list[int(u2)]),int(len(user_issue_list[int(u2)])), replace=False)
    candidate_i2_list = user_issue_list[int(u2)][candidate_i2_list]
    i_u_r_u_i.extend([(i1, u1, r, u2, i2) for i1 in candidate_i1_list for i2 in candidate_i2_list])

i_u_r_u_i = np.array(i_u_r_u_i)
i_u_r_u_i[:, 3] += len(users)
i_u_r_u_i[:, [0, 4]] += len(users) + len(repos)
sorted_index = sorted(list(range(len(i_u_r_u_i))), key=lambda i: i_u_r_u_i[i, [3, 1, 0, 2, 4]].tolist())
i_u_r_u_i = i_u_r_u_i[sorted_index]

# 3-0-1-0-3 p_u_r_u_p

p_u_r_u_p= []

for u1, r, u2 in u_r_u[0:]:

    if len(user_pr_list[int(u1)]) == 0 or len(user_pr_list[int(u2)]) == 0:
        continue

    candidate_p1_list = np.random.choice(len(user_pr_list[int(u1)]), int(len(user_pr_list[int(u1)])), replace=False)
    candidate_p2_list = np.random.choice(len(user_pr_list[int(u2)]),int(len(user_pr_list[int(u2)])), replace=False)
    candidate_p2_list = user_pr_list[int(u2)][candidate_p2_list]

    p_u_r_u_p.extend([(p1, u1, r, u2, p2) for p1 in candidate_p1_list for p2 in candidate_p2_list])

p_u_r_u_p = np.array(p_u_r_u_p)
p_u_r_u_p[:, 3]+= len(users)
p_u_r_u_p[:, [0, 4]]+= len(users)+len(repos)+len(issues)
sorted_index = sorted(list(range(len(p_u_r_u_p))), key=lambda i :p_u_r_u_p[i, [3, 1, 0, 2, 4]].tolist())
p_u_r_u_p = p_u_r_u_p[sorted_index]


# 2-1-0-1-2 i_r_u_r_i

r_u_r= []
for u, r_list in user_repo_list.items():
    for r1 in r_list:
            r_u_r.extend([(r1, u, r2) for r2 in r_list])

r_u_r = np.array(r_u_r)
r_u_r[:[0, 2]] += len(users)
sorted_index = sorted(list(range(len(r_u_r))), key=lambda i :u_r_u[i, [1, 0, 2]].tolist())
r_u_r = r_u_r[sorted_index]

i_r_u_r_i = []

for r1, u, r2 in r_u_r[0:]:

    if len(repo_issue_list[int(r1)]) == 0 or len(repo_issue_list[int(r2)]) == 0:
        continue

    candidate_i1_list = np.random.choice(len(repo_issue_list[int(r1)]), int(len(repo_issue_list[int(r1)])), replace=False)
    candidate_i2_list = np.random.choice(len(repo_issue_list[int(r2)]),int(len(repo_issue_list[int(r2)])), replace=False)
    candidate_i2_list = repo_issue_list[int(r2)][candidate_i2_list]

    i_u_r_u_i.extend([(i1, r1, u, r2, i2) for i1 in candidate_i1_list for i2 in candidate_i2_list])

i_r_u_r_i = np.array(i_r_u_r_i)
i_r_u_r_i[:, [1,3]] += len(users)
i_r_u_r_i[:, [0, 4]] += len(users) + len(repos)
sorted_index = sorted(list(range(len(i_r_u_r_i))), key=lambda i: i_r_u_r_i[i, [3, 1, 0, 2, 4]].tolist())
i_r_u_r_i = i_r_u_r_i[sorted_index]

# 3-1-0-1-3 p_r_u_r_p

p_r_u_r_p = []

for r1, u, r2 in r_u_r[0:]:

    if len(repo_pr_list[int(r1)]) == 0 or len(repo_pr_list[int(r2)]) == 0:
        continue

    candidate_p1_list = np.random.choice(len(repo_pr_list[int(r1)]), int(len(repo_pr_list[int(r1)])),
                                         replace=False)
    candidate_p2_list = np.random.choice(len(repo_pr_list[int(r2)]), int(len(repo_pr_list[int(r2)])),
                                         replace=False)
    candidate_p2_list = repo_pr_list[int(r2)][candidate_p2_list]

    p_r_u_r_p.extend([(p1, r1, u, r2, p2) for p1 in candidate_p1_list for p2 in candidate_p2_list])

p_r_u_r_p = np.array(p_r_u_r_p)
p_r_u_r_p[:, [1, 3]] += len(users)
p_r_u_r_p[:, [0, 4]] += len(users) + len(repos)+len(issues)
sorted_index = sorted(list(range(len(p_r_u_r_p))), key=lambda i: p_r_u_r_p[i, [3, 1, 0, 2, 4]].tolist())
p_r_u_r_p = p_r_u_r_p[sorted_index]



expected_metapaths = [[],[],[(2,2),(2,0,2),(2,1,2),(2,1,1,2),(2,0,1,0,2),(2,1,0,1,2)],
                      [(3,3),(3,0,3),(3,1,3),(3,1,1,3),(3,0,1,0,3),(3,1,0,1,3)]]

metapath_indices_mapping = {(2,2):i_i, (2,0,2):i_u_i, (2,1,2):i_r_i, (2,1,1,2):i_r_r_i,
                            (2,0,1,0,2):i_u_r_u_i,(2,1,0,1,2):i_r_u_r_i,
                            (3,3):p_p, (3,0,3):p_u_p, (3,1,3): p_r_p,(3,1,1,3):p_r_r_p,
                            (3,0,1,0,3):p_u_r_u_p,(3,1,0,1,3):p_r_u_r_p}



target_idx_lists = [np.arange(num_users),np.arange(num_repos),np.arange(num_issues), np.arange(num_prs)]
offset_list = [0, num_users,num_repos+num_users,num_repos+num_users+num_issues]

# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_path + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

for i, metapaths in enumerate(expected_metapaths):

    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]

        with open(save_path + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file: ###  每一个路径存一个pickle 文件
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx+ offset_list[i]:
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        np.save(save_path + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

        with open(save_path + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:

                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]

                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
                else:
                    out_file.write('{}\n'.format(target_idx))
                left = right


