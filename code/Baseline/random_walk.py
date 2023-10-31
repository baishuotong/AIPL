import argparse
import numpy as np
import networkx as nx
import random
import math, dgl
import torch as th
import pandas as pd
from dgl.data.utils import load_graphs

num_user = 7957
num_repo = 2775
num_issue = 9213
num_pr = 6349

def parse_args():
    parser = argparse.ArgumentParser(description="Just")

    parser.add_argument('--input')

    parser.add_argument('--node_types')

    parser.add_argument('--output')

    parser.add_argument('--dimensions', type=int)

    parser.add_argument('--walk_length', type=int)

    parser.add_argument('--num_walks', type=int)

    parser.add_argument('--window-size', type=int)

    parser.add_argument('--alpha', type=float)

    parser.add_argument('--workers', default=1)

    return parser.parse_args()

def save_graphs(csv_path,save_path):

    user_user = pd.read_csv(csv_path+'user_co_index.txt', sep='\t', header=None, names=['user_id', 'user_id_1'], keep_default_na=False, encoding='utf-8')
    user_issue=pd.read_csv(csv_path+'user_issue_index.txt', sep='\t', header=None, names=['user_id', 'issue_id'], keep_default_na=False, encoding='utf-8')
    user_pr=pd.read_csv('./react/index/user_pr_index_1.txt', sep='\t', header=None, names=['user_id', 'pr_id'], keep_default_na=False, encoding='utf-8')
    repo_repo = pd.read_csv('./react/index/similar_repo_index.txt', sep='\t', header=None,
                            names=['repo_id', 'repo_id_1'], keep_default_na=False, encoding='utf-8')
    repo_user = pd.read_csv('./react/index/repo_user_index.txt', sep='\t', header=None, names=['repo_id', 'user_id'], keep_default_na=False, encoding='utf-8')
    repo_issue = pd.read_csv('./react/index/repo_issue_index.txt', sep='\t', header=None, names=['repo_id', 'issue_id'],
                             keep_default_na=False, encoding='utf-8')
    repo_pr = pd.read_csv('./react/index/repo_pr_index.txt', sep='\t', header=None, names=['repo_id', 'pr_id'],
                          keep_default_na=False, encoding='utf-8')
    issue_issue = pd.read_csv('./react/index/issue_issue_index.txt', sep='\t', header=None,
                              names=['issue_id', 'issue_id_1'], keep_default_na=False, encoding='utf-8')

    pr_pr = pd.read_csv('./react/index/pr_pr.txt', sep='\t', header=None, names=['pr_id', 'pr_id_1'],
                        keep_default_na=False, encoding='utf-8')

    # issues = pd.read_csv('./react/index/issue_index.txt', sep='\t', header=None, names=['issue_id', 'issue'],
    #                      keep_default_na=False, encoding='utf-8')
    # prs = pd.read_csv('./react/index/pr_index.txt', sep='\t', header=None, names=['pr_id', 'pr'], keep_default_na=False,
    #                   encoding='utf-8')
    # # users = pd.read_csv('./react/index/user_index.txt', sep='\t', header=None, names=['user_id', 'user'], keep_default_na=False, encoding='utf-8')
    # repos = pd.read_csv('./react/index/repo_index.txt', sep='\t', header=None, names=['repo_id', 'repo'],
    #                     keep_default_na=False, encoding='utf-8')
    issue_pr = pd.read_csv('./react/index/issue_pr.txt', sep='\t', header=None, names=['issue_id', 'pr_id'],
                           keep_default_na=False, encoding='utf-8')

    graph_data = {
        # ('user', 'user_coo_user', 'user'): (th.tensor(user_user['user_id'].tolist()), th.tensor(user_user['user_id_1'].tolist())),
        ('user', 'user_propose_issue', 'issue'): (user_issue['user_id'].tolist(), user_issue['issue_id'].tolist()),
        ('user', 'user_propose_pr', 'pr'): (user_pr['user_id'].tolist(), user_pr['pr_id'].tolist()),
        # ('repo', 'repo_sim_repo', 'repo'): (th.tensor(repo_repo['repo_id'].tolist()), th.tensor(repo_repo['repo_id_1'].tolist())),
        # ('user', 'user_belong_repo', 'repo'): (th.tensor(repo_user['user_id'].tolist()),th.tensor(repo_user['repo_id'].tolist())),
        ('issue', 'issue_belong_repo', 'repo'): (repo_issue['issue_id'].tolist(), repo_issue['repo_id'].tolist()),
        ('pr', 'pr_belong_repo', 'repo'): (repo_pr['pr_id'].tolist(),repo_pr['repo_id'].tolist()),
        ('issue', 'issue_sim_issue', 'issue'): (issue_issue['issue_id'].tolist(), issue_issue['issue_id_1'].tolist()),
        ('pr', 'pr_sim_pr', 'pr'): (pr_pr['pr_id'].tolist(),pr_pr['pr_id_1'].tolist()),
        ('issue', 'issue_link_pr', 'pr'): (issue_pr['issue_id'].tolist(), issue_pr['pr_id'].tolist())
        }

    g = dgl.heterograph(graph_data)
    dgl.save_graphs(save_path,g)
    return g

# number of memorized domains = 1
def dblp_generation(G, path_length, heterg_dictionary, edge_dict, alpha, start=None):

    path = []

    path.append(start)

    cnt = 1
    homog_length = 1
    no_next_types = 0
    heterg_probability = 0

    while len(path) < path_length:
        if no_next_types == 1:
            break
              ## 节点的类型
        node=path[-1]
        cur=[k for k, v in heterg_dictionary.items() if node in v][0]

        node_types=list(heterg_dictionary.keys())
        homog_type = cur
        heterg_type = [x for x in node_types if x!=cur]

        heterg_probability = 1 - math.pow(alpha, homog_length)  ### 跳转到异质节点的概率
        r = random.uniform(0, 1)
        next_type_options = []
        if r <= heterg_probability: #### 跳跃概率
            for heterg_type_iterator in heterg_type:  ### 其他的节点
                for item in edge_dict.keys():
                    if heterg_type_iterator in item and cur in item:
                        edge_index=item
                        next_type_options.extend([e for e in edge_dict[edge_index] if node ==e[0] or node ==e[1]]) ##### 随机跳转至异质节点的列表
            if len(next_type_options)==0:
                for item in edge_dict.keys():
                    item_list=item.split('_')
                    if cur==item_list[0] and cur==item_list[2]:
                        next_type_options.extend(
                            [e for e in edge_dict[item] if node == e[0] or node == e[1]])  ##### 随机跳转至同质节点的列表

        else:
            for item in edge_dict.keys():
                item_list = item.split('_')
                if cur == item_list[0] and cur == item_list[2]:
                    next_type_options.extend(
                        [e for e in edge_dict[item] if node == e[0] or node == e[1]])  ##### 随机跳转至同质节点的列表
            if not next_type_options:
                for heterg_type_iterator in heterg_type:  ### 其他的节点
                    for item in edge_dict.keys():
                        if heterg_type_iterator in item and cur in item:
                            edge_index = item
                            next_type_options.extend(
                                [e for e in edge_dict[edge_index] if node == e[0] or node == e[1]])  ##### 随机跳转至异质节点的列表
        if not next_type_options:
            no_next_types = 1
            break
        next_node_list = []
        for item in next_type_options:
            next_node_list.extend([x for x in item if x != node])

        next_node = random.choice(next_node_list)
        path.append(next_node)
    print(path)
    return path


def generate_walks(G, num_walks, walk_length, heterg_dictionary, edge_dict,nodes):
    print('Generating walks .. ')
    walks=[]
    for cnt in range(num_walks):
        node=random.choice(nodes)
        print(node)
        just_walks = dblp_generation(G, walk_length, heterg_dictionary, edge_dict, alpha=0.05, start=node)
        walks.append(just_walks)

    return walks

def generate_node_types(G):
    heterg_dictionary = {}

    for node_type in G.ntypes:

        if node_type=='user':
            heterg_dictionary.update({node_type:list(range(0,num_user))})

        if node_type=='repo':
            heterg_dictionary.update({node_type: list(range(num_user, num_user+num_repo))})

        if node_type=='issue':
            heterg_dictionary.update({node_type: list(range(num_user+num_repo, num_user+num_repo+num_issue))})

        if node_type=='pr':
            heterg_dictionary.update({node_type: list(range(num_user+num_repo+num_issue, num_user+num_repo+num_issue+num_pr))})

    edge_dict={}
    for edge_type in G.etypes:
        edge_list=[]
        index_a = edge_type.split('_')[0]
        index_b = edge_type.split('_')[2]
        lista,listb=G.edges(etype=edge_type)

        if index_a=='repo':
            lista+=num_user
        if index_a=='issue':
            lista+=num_user+num_repo
        if index_a=='pr':
            lista+=num_user+num_repo+num_issue
        if index_b=='repo':
            listb+=num_user
        if index_b=='issue':
            listb+=num_user+num_repo
        if index_b=='pr':
            listb+=num_user+num_repo+num_issue

        for index in range(len(lista)):
            edge_list.append([lista[index],listb[index]])
        edge_dict.update({edge_type:edge_list})
    edge_dict = {key: val for key, val in edge_dict.items() if key != 'issue_link_pr'}

    return heterg_dictionary, edge_dict


def main():

    G=save_graphs(csv_path='./react/index/',save_path='./react/processed/random_walk_rgcn.bin')
    # G = nx.read_edgelist(args.input)
    # G= load_graphs('./try1.bin')
    print(G.nodes("user"))

    # heterg_dictionary, edge_dict = generate_node_types(G)
    #
    # num_walks=100
    # walk_length=20
    #
    # print ('Starting training .. ')
    #
    # print('testing')
    # acc=0
    # test_dict={}
    # test_data = pd.read_csv('./react\index\issue_pr.txt', sep='\t', header=None, names=['issue_id', 'pr_id'], keep_default_na=False, encoding='utf-8')
    #
    # issue_list = list(test_data['issue_id']+7957+2775)
    # pr_list = list(test_data['pr_id']+7957+2775+9213)
    # print(len(issue_list))
    # for index in range(len(issue_list)):
    #     test_dict.update({issue_list[index]:pr_list[index]})
    # walks = generate_walks(G, num_walks, walk_length, heterg_dictionary, edge_dict,issue_list)
    # print(len(issue_list))
    # for walk in walks:
    #     if test_dict[walk[0]] in walk[1:]:
    #         acc+=1
    #
    # print(acc/len(walks))

if __name__ == "__main__":
    main()
    # args = parse_args()
    # main(args)
