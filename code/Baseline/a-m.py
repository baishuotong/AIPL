import pandas as pd
import requests, lxml, time, json, random,traceback
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import skgarden
from skgarden import MondrianForestClassifier
from sklearn.metrics import precision_score, roc_curve, recall_score, f1_score, roc_auc_score, accuracy_score

headers = {"Authorization": "token " + "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",
           'Accept': 'application/vnd.github.v3+json',
           'User-Agent': 'Mozilla/5.0', }

data_df=pd.read_csv('./Data/vue_a_m.csv',encoding='utf-8')
data_df_neg=pd.read_csv('./Data/vue_a_m_neg.csv', encoding='utf-8')


def PR_information():

    # data_df['p_diff']=''
    # data_df['lack_of_description']=''
    # data_df['p_commit']=''

    for index in range(0,len(data_df)):
        # print(index)
        if data_df['p_diff'].loc[index]>=0:
            # print(data_df['p_diff'].loc[index])
            data_df['p_diff'].loc[index] = data_df['p_diff'].loc[index]
            # data_df.to_csv('./react/react_am_3.csv', index=False)
        else:
            try:
                ##### lack of description
                # body_len=len([x for x in data_df['target_body'].loc[index].split(' ') if x!=''])
                # if body_len>1:
                #     data_df['lack_of_description'].loc[index]=1
                # else:
                #     data_df['lack_of_description'].loc[index]=0

                ######  Size of branch(p.commit)
                # pr_commit_url='https://api.github.com/repos/'+data_df['target_repo'].loc[index]+'/pulls/'+str(data_df['target_num'].loc[index])+'/commits'
                # print(pr_commit_url)
                # commits=requests.get(pr_commit_url, headers=headers)
                # commits_json = commits.json()
                # commits.close()
                # if 'sha' not in commits_json[0].keys():
                #     print(commits_json)
                #     print(data_df['target_repo'].loc[index])
                # p_commit.append(len(commits_json))

                # ##### Number of files touched(p.diff)
                # repo=test_list[index].split('_')[0]
                # print(repo)
                # num=str(test_list[index].split('_')[1])
                print(data_df['p_diff'].loc[index] )
                pr_diff_url='https://patch-diff.githubusercontent.com/raw/'+data_df['target_repo'].loc[index]+'/pull/'+str(data_df['target_num'].loc[index])+'.diff'
                print(pr_diff_url)
                diffs=requests.get(pr_diff_url, headers=headers)
                pr_diff = diffs.text
                diffs.close()
                # diff_list.update({test_list[index]:pr_diff.count('diff --')})
                # json_str = json.dumps(diff_list)
                # with open('./Data/test_data_1.json', 'w') as json_file:
                #     json_file.write(json_str)
                data_df['p_diff'].loc[index]=pr_diff.count('diff --')
                # data_df['p_commit'].loc[index]= len(commits_json)
                # print(data_df['lack_of_description'].loc[index],len(commits_json),pr_diff.count('diff --'))
                data_df.to_csv('./react/react_am_3.csv', index=False)
            except:
                traceback.print_exc()
                print(index)
                break
                time.sleep(120)
            # if len(data_df['target_body'].loc[index])>1:
            #     data_df['lack_of_description'].loc[index]=1
            # else:
            #     data_df['lack_of_description'].loc[index]=0
            # data_df['p_diff'].loc[index] = 0
            # data_df['p_commit'].loc[index] = 0
            # # time.sleep(120)



def issue_information():
    report_size = []
    existing_links=[]
    reopens=[]
    partipants=[]
    # data_df['report_size'], data_df['participants'], data_df['reopens'], data_df['existing_links']=[0,0,0,0]

    for index in range(0, 3750):
        ###### report size
        report=data_df['source_body'].loc[index]
        report=[x for x in report.split(' ') if x!='']
        report_size.append(len(report))

        ####### Participants

        issue_url='https://api.github.com/repos/'+data_df['source_repo'].loc[index].replace(' ','/')+'/issues/'+str(data_df['source_num'].loc[index])
        issue = requests.get(issue_url, headers=headers)
        issue_json = issue.json()
        issue.close()
        print(issue_url)
        user=issue_json['user']['login']
        per_participant=[]
        another_page = True
        comments_results = []
        num = 0
        while another_page:
            num += 1
            issue_comment_url = 'https://api.github.com/repos/'+data_df['source_repo'].loc[index].replace(' ','/')+'/issues/'+str(data_df['source_num'].loc[index])+'/comments?page='\
                                + str(num) + "&per_page=100"
            print(issue_comment_url)
            issue_comment = requests.get(issue_comment_url, headers=headers)
            issue_comment_json = issue_comment.json()
            issue_comment.close()
            comments_results.extend(issue_comment_json)
            if 'next' in issue_comment.links:
                timeline_url = issue_comment.links['next']
            else:
                another_page = False
        for comment in comments_results:
            per_participant.append(comment['user']['login'])

        partipants.append(len([x for x in per_participant if x == user]))

        ####### reopens

        another_page = True
        timelines_results = []
        num = 0
        timeline_flag=0
        while another_page:
            num += 1
            issue_timeline_url = 'https://api.github.com/repos/'+data_df['source_repo'].loc[index].replace(' ','/')+'/issues/'+str(data_df['source_num'].loc[index])+'/timeline?page='\
                                + str(num) + "&per_page=100"
            issue_timeline = requests.get(issue_timeline_url, headers=headers)
            issue_timeline_json = issue_timeline.json()
            issue_timeline.close()
            timelines_results.extend(issue_timeline_json)
            if 'next' in issue_timeline.links:
                timeline_url = issue_timeline.links['next']
            else:
                another_page = False
        for timeline in timelines_results:
            if timeline['event']=='closed' and timelines_results.index(timeline)<len(timelines_results)-1:
                timeline_flag+=1

        reopens.append(timeline_flag)

        ##### existing_links
        existing_links.append(1)

        data_df['report_size'].loc[index] = len(report)
        data_df['existing_links'].loc[index] = 1
        data_df['reopens'].loc[index] = timeline_flag
        data_df['participants'].loc[index] = len([x for x in per_participant if x == user])

        print(index)

        data_df.to_csv('./react/react_a_m.csv', index=False)

def tf_idf(corpus):
    tfidf_vec = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=0.0001)
    tfidf_matrix = tfidf_vec.fit_transform(corpus)
    tf = tfidf_vec.transform(corpus).toarray()

    return tf

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim

def cs(tf,corpus):
    cs_full_list=[]
    cs_tt_list=[]
    cs_bb_list=[]
    cs_sb_tt_list=[]
    cs_st_tb_list=[]
    for index in range(len(data_df)):
        index_1 = corpus.index(data_df['source_body'].loc[index])
        index_2 = corpus.index(data_df['target_body'].loc[index])
        v1 = tf[index_1]
        v2 = tf[index_2]
        cs_full=cos_sim(v1,v2)     ##### cs_full 的值

        index_3 = corpus.index(data_df['source_title'].loc[index])
        index_4 = corpus.index(data_df['target_title'].loc[index])
        v3 = tf[index_3]
        v4 = tf[index_4]
        cs_t_t=cos_sim(v3, v4)    ##### cs_title 的值

        index_5 = corpus.index(data_df['source_body'].loc[index])-corpus.index(data_df['source_title'].loc[index])
        index_6 = corpus.index(data_df['target_body'].loc[index])-corpus.index(data_df['target_title'].loc[index])
        v5 = tf[index_5]
        v6 = tf[index_6]
        cs_b_b=cos_sim(v5, v6)    ##### CS description-report 的值

        index_7 = corpus.index(data_df['source_body'].loc[index])-corpus.index(data_df['source_title'].loc[index])
        index_8 = corpus.index(data_df['target_title'].loc[index])
        v7 = tf[index_7]
        v8 = tf[index_8]
        cs_sb_tt = cos_sim(v7, v8)    ##### CS p.title,i.report 的值

        index_9 = corpus.index(data_df['source_title'].loc[index])
        index_10 = corpus.index(data_df['target_body'].loc[index])-corpus.index(data_df['target_title'].loc[index])
        v9 = tf[index_9]
        v10 = tf[index_10]
        cs_st_tb = cos_sim(v9, v10)    ##### CS p.description,i.title 的值

        cs_full_list.append(cs_full)
        cs_tt_list.append(cs_t_t)
        cs_bb_list.append(cs_b_b)
        cs_sb_tt_list.append(cs_sb_tt)
        cs_st_tb_list.append(cs_st_tb)

    data_df['cs_full']=pd.Series(cs_full_list)
    data_df['cs_tt_list'] = pd.Series(cs_tt_list)
    data_df['cs_bb_list'] = pd.Series(cs_bb_list)
    data_df['cs_sb_tt_list'] = pd.Series(cs_sb_tt_list)
    data_df['cs_st_tb_list']= pd.Series(cs_st_tb_list)

def Jaccrad_similarity(sentence_A, sentence_B):  # terms_reference为源句子，terms_model为候选句子
    sentence_A = [x for x in sentence_A.split(' ') if x!='']  # 默认精准模式
    sentence_B = [x for x in sentence_B.split(' ') if x!='']
    grams_A = set(sentence_A)  # 去重；如果不需要就改为list
    grams_B = set(sentence_B)
    temp = 0
    for i in grams_A:
        if i in grams_B:
            temp = temp + 1
    fenmu = len(grams_A) + len(grams_B) - temp  # 并集
    if  temp>0:
        jaccard_coefficient = float(temp / fenmu)  # 交集
    else:
        jaccard_coefficient=0
    return jaccard_coefficient

def Js():

    js_full_list=[]
    js_tt_list=[]
    js_bb_list=[]
    js_sb_tt_list=[]
    js_st_tb_list=[]

    for index in range(len(data_df)):
        sentence_a = data_df['source_body'].loc[index]
        sentence_b = data_df['target_body'].loc[index]
        js_full = Jaccrad_similarity(sentence_a, sentence_b)  ##### js_full 的值

        sentence_c = data_df['source_title'].loc[index]
        sentence_d = data_df['source_body'].loc[index].replace(data_df['source_title'].loc[index],'')

        sentence_e = data_df['target_title'].loc[index]
        sentence_f = data_df['target_body'].loc[index].replace(data_df['target_title'].loc[index], '')

        js_t_t = Jaccrad_similarity(sentence_c, sentence_e)  ##### js_title 的值
        js_b_b = Jaccrad_similarity(sentence_d, sentence_f)  ##### js description-report 的值
        js_sb_tt = Jaccrad_similarity(sentence_d, sentence_e)  ##### js p.title,i.report 的值
        js_st_tb = Jaccrad_similarity(sentence_c, sentence_f)  ##### jS p.description,i.title 的值

        js_full_list.append(js_full)
        js_tt_list.append(js_t_t)
        js_bb_list.append(js_b_b)
        js_sb_tt_list.append(js_sb_tt)
        js_st_tb_list.append(js_st_tb)

    data_df['js_full'] = pd.Series(js_full_list)
    data_df['js_tt_list'] = pd.Series(js_tt_list)
    data_df['js_bb_list'] = pd.Series(js_bb_list)
    data_df['js_sb_tt_list'] = pd.Series(js_sb_tt_list)
    data_df['js_st_tb_list'] = pd.Series(js_st_tb_list)

def Dice_similarity(sentence_A, sentence_B):  # terms_reference为源句子，terms_model为候选句子
    sentence_A = [x for x in sentence_A.split(' ') if x!='']  # 默认精准模式
    sentence_B = [x for x in sentence_B.split(' ') if x!='']
    grams_A = set(sentence_A)  # 去重；如果不需要就改为list
    grams_B = set(sentence_B)
    temp = 0
    for i in grams_A:
        if i in grams_B:
            temp = temp + 1
    fenmu = min(len(grams_A), len(grams_B)) # 取最小值
    if fenmu!=0:
        d_coefficient = float(temp / fenmu)  # 交集
    else:
        d_coefficient = 0
    return d_coefficient

def Ds():

    ds_full_list=[]
    ds_tt_list=[]
    ds_bb_list=[]
    ds_sb_tt_list=[]
    ds_st_tb_list=[]

    for index in range(len(data_df)):
        sentence_a =data_df['source_body'].loc[index]
        sentence_b =data_df['target_body'].loc[index]
        ds_full = Dice_similarity(sentence_a, sentence_b)  ##### ds_full 的值

        sentence_c = data_df['source_title'].loc[index]
        sentence_d = data_df['source_body'].loc[index].replace(data_df['source_title'].loc[index], '')

        sentence_e = data_df['target_title'].loc[index]
        sentence_f = data_df['target_body'].loc[index].replace(data_df['target_title'].loc[index], '')

        ds_t_t = Dice_similarity(sentence_c, sentence_e)  ##### ds_title 的值
        ds_b_b = Dice_similarity(sentence_d, sentence_f)  ##### ds description-report 的值
        ds_sb_tt = Dice_similarity(sentence_d, sentence_e)  ##### ds p.title,i.report 的值
        ds_st_tb = Dice_similarity(sentence_c, sentence_f)  ##### dS p.description,i.title 的值

        ds_full_list.append(ds_full)
        ds_tt_list.append(ds_t_t)
        ds_bb_list.append(ds_b_b)
        ds_sb_tt_list.append(ds_sb_tt)
        ds_st_tb_list.append(ds_st_tb)

    data_df['ds_full'] = pd.Series(ds_full_list)
    data_df['ds_tt_list'] = pd.Series(ds_tt_list)
    data_df['ds_bb_list'] = pd.Series(ds_bb_list)
    data_df['ds_sb_tt_list'] = pd.Series(ds_sb_tt_list)
    data_df['ds_st_tb_list'] = pd.Series(ds_st_tb_list)

def construct_neg_dataset():
    #### 构造neg数据集
    data_df =pd.read_csv('./react/react_a_m.csv', encoding='utf-8')
    for index in range(len(data_df)):
        random_list=[]
        for i in range(len(data_df)):
            if data_df['source_body'].loc[index]!=data_df['source_body'].loc[i]:
                random_list.append(i)
        j=random.sample(random_list,1)[0]
        # f.write(data_df['source_repo'].loc[index]+'\t'+str(data_df['source_num'].loc[index])+'\t'+data_df['source_title'].loc[index]+'\t'+data_df['source_body'].
        #         loc[index]+'\t'+str(data_df['report_size'].loc[index])+'\t'+str(data_df['participants'].loc[index])+'\t'+str(data_df['reopens'].loc[index])+'\t'+
        #         str(data_df['existing_links'].loc[index])+'\t'+data_df['target_repo'].loc[j]+'\t'+str(data_df['target_num'].loc[j])+'\t'+
        #         data_df['target_title'].loc[j]+'\t'+data_df['target_body'].loc[j]+'\t'+str(data_df['lack_of_description'].loc[j])+'\t'+
        #         str(data_df['p_commit'].loc[j])+'\t'+str(data_df['p_diff'].loc[j])+'\n')
        data_df['target_repo'].loc[index]=data_df['target_repo'].loc[j]
        data_df['target_num'].loc[index]=data_df['target_num'].loc[j]
        data_df['target_title'].loc[index]=data_df['target_title'].loc[j]
        data_df['target_body'].loc[index] = data_df['target_body'].loc[j]
        data_df['lack_of_description'].loc[index]= data_df['lack_of_description'].loc[j]
        data_df['p_commit'].loc[index] = data_df['p_commit'].loc[j]
        data_df['p_commit'].loc[index] = data_df['p_diff'].loc[j]
        data_df.to_csv('./react/react_a_m_neg.csv', index=False)

def construct_train_dataset():
    full_list=[x for x in range(len(data_df))]
    random.shuffle(full_list)
    train_ids=full_list[:int(len(full_list)*0.8)]
    test_ids=full_list[int(len(full_list)*0.8):]
    X_train, y_train=[],[]
    X_test, y_test = [], []
    for train_index in train_ids:
        X_train.append(np.nan_to_num(np.array([data_df['report_size'].loc[train_index], data_df['participants'].loc[train_index], data_df['reopens'].loc[train_index],
                                data_df['existing_links'].loc[train_index], data_df['existing_links'].loc[train_index], data_df['lack_of_description'].loc[train_index],
                                 data_df['p_commit'].loc[train_index], data_df['p_diff'].loc[train_index], data_df['cs_full'].loc[train_index], data_df['cs_tt_list'].loc[train_index],
                                 data_df['cs_bb_list'].loc[train_index], data_df['cs_sb_tt_list'].loc[train_index], data_df['cs_st_tb_list'].loc[train_index],
                                 data_df['js_full'].loc[train_index], data_df['js_tt_list'].loc[train_index], data_df['js_bb_list'].loc[train_index], data_df['js_sb_tt_list'].loc[train_index],
                                 data_df['js_st_tb_list'].loc[train_index], data_df['ds_full'].loc[train_index], data_df['ds_tt_list'].loc[train_index], data_df['ds_bb_list'].loc[train_index],
                                 data_df['ds_sb_tt_list'].loc[train_index], data_df['ds_st_tb_list'].loc[train_index]])))
        y_train.append(1)
        X_train.append(np.nan_to_num(np.array([data_df_neg['report_size'].loc[train_index], data_df_neg['participants'].loc[train_index], data_df_neg['reopens'].loc[train_index],
                                data_df_neg['existing_links'].loc[train_index], data_df_neg['existing_links'].loc[train_index], data_df_neg['lack_of_description'].loc[train_index],
                                 data_df_neg['p_commit'].loc[train_index], data_df_neg['p_diff'].loc[train_index], data_df_neg['cs_full'].loc[train_index], data_df_neg['cs_tt_list'].loc[train_index],
                                 data_df_neg['cs_bb_list'].loc[train_index], data_df_neg['cs_sb_tt_list'].loc[train_index], data_df_neg['cs_st_tb_list'].loc[train_index],
                                 data_df_neg['js_full'].loc[train_index], data_df_neg['js_tt_list'].loc[train_index], data_df_neg['js_bb_list'].loc[train_index], data_df_neg['js_sb_tt_list'].loc[train_index],
                                 data_df_neg['js_st_tb_list'].loc[train_index], data_df_neg['ds_full'].loc[train_index], data_df_neg['ds_tt_list'].loc[train_index], data_df_neg['ds_bb_list'].loc[train_index],
                                 data_df_neg['ds_sb_tt_list'].loc[train_index], data_df_neg['ds_st_tb_list'].loc[train_index]])))
        y_train.append(0)
    for test_index in test_ids:
        X_test.append(np.nan_to_num(np.array([data_df['report_size'].loc[test_index], data_df['participants'].loc[test_index], data_df['reopens'].loc[test_index],
                                data_df['existing_links'].loc[test_index], data_df['existing_links'].loc[test_index], data_df['lack_of_description'].loc[test_index],
                                 data_df['p_commit'].loc[test_index], data_df['p_diff'].loc[test_index], data_df['cs_full'].loc[test_index], data_df['cs_tt_list'].loc[test_index],
                                 data_df['cs_bb_list'].loc[test_index], data_df['cs_sb_tt_list'].loc[test_index], data_df['cs_st_tb_list'].loc[test_index],
                                 data_df['js_full'].loc[test_index], data_df['js_tt_list'].loc[test_index], data_df['js_bb_list'].loc[test_index], data_df['js_sb_tt_list'].loc[test_index],
                                 data_df['js_st_tb_list'].loc[test_index], data_df['ds_full'].loc[test_index], data_df['ds_tt_list'].loc[test_index], data_df['ds_bb_list'].loc[test_index],
                                 data_df['ds_sb_tt_list'].loc[test_index], data_df['ds_st_tb_list'].loc[test_index]])))
        y_test.append(1)
        X_test.append(np.nan_to_num(np.array([data_df_neg['report_size'].loc[test_index], data_df_neg['participants'].loc[test_index], data_df_neg['reopens'].loc[test_index],
                                data_df_neg['existing_links'].loc[test_index], data_df_neg['existing_links'].loc[test_index], data_df_neg['lack_of_description'].loc[test_index],
                                 data_df_neg['p_commit'].loc[test_index], data_df_neg['p_diff'].loc[test_index], data_df_neg['cs_full'].loc[test_index], data_df_neg['cs_tt_list'].loc[test_index],
                                 data_df_neg['cs_bb_list'].loc[test_index], data_df_neg['cs_sb_tt_list'].loc[test_index], data_df_neg['cs_st_tb_list'].loc[test_index],
                                 data_df_neg['js_full'].loc[test_index], data_df_neg['js_tt_list'].loc[test_index], data_df_neg['js_bb_list'].loc[test_index], data_df_neg['js_sb_tt_list'].loc[test_index],
                                 data_df_neg['js_st_tb_list'].loc[test_index], data_df_neg['ds_full'].loc[test_index], data_df_neg['ds_tt_list'].loc[test_index], data_df_neg['ds_bb_list'].loc[test_index],
                                 data_df_neg['ds_sb_tt_list'].loc[test_index], data_df_neg['ds_st_tb_list'].loc[test_index]])))
        y_test.append(0)

    return X_train, y_train, X_test, y_test

def construct_train_recent_dataset(recent_index_path, data_df, data_df_neg):

    recent_test_set= pd.read_csv(recent_index_path, sep='\t', header=None, names=['issue_id', 'issue', 'pr_id','pr'],
                           keep_default_na=False, encoding='utf-8')

    test_index=[]

    for index in range(len(recent_test_set)):

        for i in range(len(data_df)):

            if data_df['source_repo'].loc[i].replace(' ','/')==recent_test_set['issue'].loc[index].split('__')[0] and data_df['source_num'].loc[i]==int(recent_test_set['issue'].loc[index].split('__')[1]):

                if data_df['target_repo'].loc[i].replace(' ','/')==recent_test_set['pr'].loc[index].split('__')[0] and data_df['target_num'].loc[i]==int(recent_test_set['pr'].loc[index].split('__')[1]):
                    test_index.append(i)

    train_list=[x for x in range(len(data_df)) if x not in test_index]
    random.shuffle(train_list)
    # train_ids=train_list[:int(len(train_list)*0.8)]
    X_train, y_train=[],[]
    X_test, y_test = [], []

    for train_index in train_list:
        X_train.append(np.nan_to_num(np.array([data_df['report_size'].loc[train_index], data_df['participants'].loc[train_index], data_df['reopens'].loc[train_index],
                                data_df['existing_links'].loc[train_index], data_df['existing_links'].loc[train_index], data_df['lack_of_description'].loc[train_index],
                                 data_df['p_commit'].loc[train_index], data_df['p_diff'].loc[train_index], data_df['cs_full'].loc[train_index], data_df['cs_tt_list'].loc[train_index],
                                 data_df['cs_bb_list'].loc[train_index], data_df['cs_sb_tt_list'].loc[train_index], data_df['cs_st_tb_list'].loc[train_index],
                                 data_df['js_full'].loc[train_index], data_df['js_tt_list'].loc[train_index], data_df['js_bb_list'].loc[train_index], data_df['js_sb_tt_list'].loc[train_index],
                                 data_df['js_st_tb_list'].loc[train_index], data_df['ds_full'].loc[train_index], data_df['ds_tt_list'].loc[train_index], data_df['ds_bb_list'].loc[train_index],
                                 data_df['ds_sb_tt_list'].loc[train_index], data_df['ds_st_tb_list'].loc[train_index]])))
        y_train.append(1)
        X_train.append(np.nan_to_num(np.array([data_df_neg['report_size'].loc[train_index], data_df_neg['participants'].loc[train_index], data_df_neg['reopens'].loc[train_index],
                                data_df_neg['existing_links'].loc[train_index], data_df_neg['existing_links'].loc[train_index], data_df_neg['lack_of_description'].loc[train_index],
                                 data_df_neg['p_commit'].loc[train_index], data_df_neg['p_diff'].loc[train_index], data_df_neg['cs_full'].loc[train_index], data_df_neg['cs_tt_list'].loc[train_index],
                                 data_df_neg['cs_bb_list'].loc[train_index], data_df_neg['cs_sb_tt_list'].loc[train_index], data_df_neg['cs_st_tb_list'].loc[train_index],
                                 data_df_neg['js_full'].loc[train_index], data_df_neg['js_tt_list'].loc[train_index], data_df_neg['js_bb_list'].loc[train_index], data_df_neg['js_sb_tt_list'].loc[train_index],
                                 data_df_neg['js_st_tb_list'].loc[train_index], data_df_neg['ds_full'].loc[train_index], data_df_neg['ds_tt_list'].loc[train_index], data_df_neg['ds_bb_list'].loc[train_index],
                                 data_df_neg['ds_sb_tt_list'].loc[train_index], data_df_neg['ds_st_tb_list'].loc[train_index]])))
        y_train.append(0)

    for test_index in test_index:
        X_test.append(np.nan_to_num(np.array([data_df['report_size'].loc[test_index], data_df['participants'].loc[test_index], data_df['reopens'].loc[test_index],
                                data_df['existing_links'].loc[test_index], data_df['existing_links'].loc[test_index], data_df['lack_of_description'].loc[test_index],
                                 data_df['p_commit'].loc[test_index], data_df['p_diff'].loc[test_index], data_df['cs_full'].loc[test_index], data_df['cs_tt_list'].loc[test_index],
                                 data_df['cs_bb_list'].loc[test_index], data_df['cs_sb_tt_list'].loc[test_index], data_df['cs_st_tb_list'].loc[test_index],
                                 data_df['js_full'].loc[test_index], data_df['js_tt_list'].loc[test_index], data_df['js_bb_list'].loc[test_index], data_df['js_sb_tt_list'].loc[test_index],
                                 data_df['js_st_tb_list'].loc[test_index], data_df['ds_full'].loc[test_index], data_df['ds_tt_list'].loc[test_index], data_df['ds_bb_list'].loc[test_index],
                                 data_df['ds_sb_tt_list'].loc[test_index], data_df['ds_st_tb_list'].loc[test_index]])))
        y_test.append(1)
        X_test.append(np.nan_to_num(np.array([data_df_neg['report_size'].loc[test_index], data_df_neg['participants'].loc[test_index], data_df_neg['reopens'].loc[test_index],
                                data_df_neg['existing_links'].loc[test_index], data_df_neg['existing_links'].loc[test_index], data_df_neg['lack_of_description'].loc[test_index],
                                 data_df_neg['p_commit'].loc[test_index], data_df_neg['p_diff'].loc[test_index], data_df_neg['cs_full'].loc[test_index], data_df_neg['cs_tt_list'].loc[test_index],
                                 data_df_neg['cs_bb_list'].loc[test_index], data_df_neg['cs_sb_tt_list'].loc[test_index], data_df_neg['cs_st_tb_list'].loc[test_index],
                                 data_df_neg['js_full'].loc[test_index], data_df_neg['js_tt_list'].loc[test_index], data_df_neg['js_bb_list'].loc[test_index], data_df_neg['js_sb_tt_list'].loc[test_index],
                                 data_df_neg['js_st_tb_list'].loc[test_index], data_df_neg['ds_full'].loc[test_index], data_df_neg['ds_tt_list'].loc[test_index], data_df_neg['ds_bb_list'].loc[test_index],
                                 data_df_neg['ds_sb_tt_list'].loc[test_index], data_df_neg['ds_st_tb_list'].loc[test_index]])))
        y_test.append(0)
    print(len(X_test))
    return X_train, y_train, X_test, y_test
def decision_tree():

    X_train, y_train, X_test, y_test=construct_train_recent_dataset(recent_index_path='./Data/index/recent_index.txt', data_df=data_df, data_df_neg=data_df_neg)
    # clf = tree.DecisionTreeClassifier()  # 实例化
    # clf=RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    clf=MondrianForestClassifier(n_estimators=16, )

    clf = clf.fit(X_train, y_train)  # 用训练集数据训练模型
    y_pred=clf.predict(X_test)
    print(type(y_pred))
    np.save(r"./Data/am_results.npy", y_pred.astype(int))
    print('accuracy:{:.3}, precision:{:.3}, recall:{:.3}, f1:{:.3}'.format(clf.score(X_test, y_test),
                                                                                       precision_score(y_test,y_pred),
                                                                                       recall_score(y_test,y_pred), f1_score(y_test,y_pred)))
#
# corpus=[]
# corpus.extend(data_df['source_title'].tolist())
# corpus.extend(data_df['target_title'].tolist())
# corpus.extend(data_df['source_body'].tolist())
# corpus.extend(data_df['target_body'].tolist())
# for index in range(len(data_df)):
#     corpus.append(data_df['source_body'].loc[index].replace(data_df['source_title'].loc[index],''))
#     corpus.append(data_df['target_body'].loc[index].replace(data_df['target_title'].loc[index], ''))
# corpus.extend(data_df['source_title'])
# corpus.extend(data_df['target_title'])
# for index in range(len(data_df)):
#     corpus.append(data_df['source_title'].loc[index]+' '+data_df['source_body'].loc[index])
#     corpus.append(data_df['target_title'].loc[index]+' '+data_df['target_body'].loc[index])
# PR_information()
# issue_information()
# tf=tf_idf(corpus)
# cs(tf,corpus)
# Js()
# Ds()
# data_df.to_csv('./react/react_a_m_neg.csv', index=False)
# construct_neg_dataset()

decision_tree()

