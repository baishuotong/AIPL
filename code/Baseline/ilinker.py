import pandas as pd
import requests,re, emoji, random, json,nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df=pd.read_csv('./react/react_info_clean_1.csv', encoding='utf-8') ## 读取csv

word_map = {"i'll": "i will", "it'll": "it will","we'll": "we will","he'll": "he will","they'll": "they will","i'd": "i would","we'd": "we would","he'd": "he would","weren't":"were not",
    "they'd": "they would","i'm": "i am","he's": "he is","she's": "she is","that's": "that is", "here's": "here is","there's": "there is","we're": "we are","won't":"will not",
    "they're": "they are","who's": "who is","what's": "what is","i've": "i have","we've": "we have","they've": "they have","wanna": "want to","can't": "can not","shouldn't":"should not",
    "ain't": "are not", "isn't": "is not","it's":"it is","doesn't":"does not","haven't":"have not","don't":"do not","you're":"you are","let's":"let","you've": "you have","he've":"he have","she've":"she have","you'd":"you would","didn't":"did not"}

def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
        if tag.startswith('NN'):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
        else:
            yield word
def clean_str(origin_str):
    if type(origin_str)==float:
        origin_str='    '
    str=''
    origin_str=origin_str.replace('\n','').replace('\t','').replace('\r','').replace('-',' ').replace('.',' ').replace('—',' ').replace('”',' ').replace('/',' ').replace('`',' ')
    #0) 还原缩写
    for word in origin_str.lower().split(' '):
        if word in word_map.keys():
            word = word_map[word]
        str += word + ' '
    #1) 过滤网址
    str=re.sub('(https?://[a-zA-Z0-9\.\?/%-_]*)',' ',str)

    #2) 过滤符号
    r = "[0-9_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/]"
    str = re.sub(r, ' ', str)

    #3) 去除emoji
    str = emoji.demojize(str)
    # str=re.sub('(\:.*?\:)',' ',str)

    #4) 去除非英文书写的文本
    re_nonenglish = re.compile(r'[^\x00-\x7f]')

    if not re_nonenglish.search(str):
        # str=str.split(' ')
        # str=[x for x in str if x!=' ' and x!='']
        # str=' '.join(str)
        str=str
    else:
        print(origin_str)
        print(str)
        str=' '
    # clean_repo_description.update({repo:str})
    #5) 词性还原
    final_str=' '.join(lemmatize_all(str.lower()))

    return final_str


def crawl_body():
    #### 选择issue_pr的index
    headers = {"Authorization": "token " + "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",
               'Accept': 'application/vnd.github.v3+json',
               'User-Agent': 'Mozilla/5.0', }
    f = open('./Data/ilinker_data.txt', 'a', encoding='utf-8')
    for index,item in enumerate(df['link_type']):
        if item=='ISS to PR' and index>4046:
            print(index)
            source_repo=df['source_repo'].loc[index]
            source_num=df['source_num'].loc[index]
            source_title=df['source_title'].loc[index]

            target_repo=df['target_repo'].loc[index]
            target_num=df['target_num'].loc[index]
            target_title=df['target_title'].loc[index]

            source_url = 'https://api.github.com/repos/'+source_repo+'/issues/' + str(source_num)
            source = requests.get(source_url, headers=headers)
            source_json = source.json()
            source.close()
            source_body=source_json['body']
            if source_body:
                source_body = source_body.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            else:
                source_body = ' '

            target_url = 'https://api.github.com/repos/'+target_repo+'/issues/' + str(target_num)
            target = requests.get(target_url, headers=headers)
            target_json = target.json()
            target.close()
            target_body = target_json['body']
            if target_body:
                target_body = target_body.replace('\n', ' ').replace('\r', ' ').replace('\t',' ')
            else:
                target_body=' '

            f.write(source_repo+'\t'+str(source_num)+'\t'+source_title+'\t'+source_body+'\t'+target_repo+'\t'+str(target_num)+'\t'+target_title+'\t'+target_body+'\n')

def crawl_issue_body(repo,num):
    #### 选择issue_pr的index
    headers = {"Authorization": "token " + "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",
               'Accept': 'application/vnd.github.v3+json',
               'User-Agent': 'Mozilla/5.0', }


    source_url = 'https://api.github.com/repos/'+repo+'/issues/' + str(num)
    source = requests.get(source_url, headers=headers)
    source_json = source.json()
    source.close()
    source_body=source_json['body']
    if source_body:
        source_body = source_body.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    else:
        source_body = ' '

    return source_body

###### 已经爬取完body 进行寻找
def find_body():
    f = open('./react/ilinker_data.txt', 'a', encoding='utf-8')
    # df_2=pd.read_csv('F:\BST\白烁彤论文\论文3\数据/facebook_react_allbody.csv', encoding='utf-8') ## 读取csv
    with open('./react/body_dict.json', 'r', encoding='utf-8') as f_2:
        body = json.load(f_2, encoding='uft-8')
    print(len(body))
    for index,item in enumerate(df['link_type']):
        if item == 'ISS to PR' and index>10178:
            print(index)
            source_repo=df['source_repo'].loc[index]
            source_num=df['source_num'].loc[index]
            source_title=df['source_title'].loc[index]

            target_repo=df['target_repo'].loc[index]
            target_num=df['target_num'].loc[index]
            target_title=df['target_title'].loc[index]

            if source_repo+'__'+str(source_num) in body.keys():
                source_body=body[source_repo+'__'+str(source_num)]
            else:
                print('need crawl')
                print(source_repo+'__'+str(source_num))
                source_body=crawl_issue_body(source_repo,source_num)

            if target_repo+'__'+str(target_num) in body.keys():
                target_body = body[target_repo + '__' + str(target_num)]
            else:
                print('need crawl')
                print(target_repo+'__'+str(target_num))
                target_body = crawl_issue_body(target_repo, target_num)

            source_body=clean_str(source_body)
            target_body=clean_str(target_body)

            body.update(
                {df['source_repo'].loc[index] + '__' + str(df['source_num'].loc[index]):source_body})
            body.update(
                {df['target_repo'].loc[index] + '__' + str(df['target_num'].loc[index]):target_body})

            f.write(source_repo + '\t' + str(
                source_num) + '\t' + source_title + '\t' + source_body +'\t'+ target_repo + '\t' + str(
                target_num) + '\t' + target_title + '\t' + target_body +'\n')

def data_processing():

    # body_text = pd.read_csv('./Data/ilinker_data.txt', sep='\t', header=None, names=['source_repo', 'source_num', 'source_title','source_body', 'target_repo',
    #                                                                                   'target_num', 'target_title','target_body'],
    #                       keep_default_na=False, encoding='utf-8')
    body_text = pd.read_csv('./Data/vue_ilinker.csv', encoding='utf-8')
    for index in range(len(body_text)):

        print(index)
        body_text['source_body'].loc[index] = body_text['source_title'].loc[index]+' '+clean_str(body_text['source_body'].loc[index])
        body_text['target_body'].loc[index] = body_text['target_title'].loc[index]+' '+clean_str(body_text['target_body'].loc[index])

    return body_text

def data_processing_2():

    body_text = pd.read_csv('./Data/ilinker_data_neg.txt', sep=',', header=None, names=['source_repo', 'source_num',
                                                                                          'source_body', 'target_repo', 'target_num','target_body'],
                          keep_default_na=False, encoding='utf-8')

    for index in range(len(body_text)):

        body_text['source_body'].loc[index] = clean_str(body_text['source_body'].loc[index])
        body_text['target_body'].loc[index] = clean_str(body_text['target_body'].loc[index])

    return body_text

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

def tf_idf_similarity(body_text,corpus):

    tf_sims=[]
    tfidf_vec = TfidfVectorizer(stop_words = 'english', max_df=0.5, min_df=0.001)
    tfidf_matrix = tfidf_vec.fit_transform(corpus)
    tf = tfidf_vec.transform(corpus).toarray()

    for index in range(len(body_text)):

        index_1=corpus.index(body_text['source_body'].loc[index])
        index_2 = corpus.index(body_text['target_body'].loc[index])
        v1=tf[index_1]
        v2=tf[index_2]
        tf_idf_sim=cos_sim(v1,v2)
        tf_sims.append(tf_idf_sim)

    return tf_sims

def training_word2vec(corpus):
    ### 训练词向量
    # sentences=word2vec.BrownCorpus(corpus)
    model = word2vec.Word2Vec(corpus, vector_size=256,min_count=2,window=5,sg=1)
    model.save('./model/word2vec_256.model')

def word2vec_sim():  #将每一句话都存成用word2vec计算的向量
    word2vec_sim_list=[]
    ### 加载单词向量模型
    model = word2vec.Word2Vec.load('./model/word2vec_256.model')
    for index in range(len(body_text)):
        source_body=body_text['source_body'].loc[index]
        target_body=body_text['target_body'].loc[index]

        source_body=[word for word in source_body.split(' ') if word in model.wv]
        target_body=[word for word in target_body.split(' ') if word in model.wv]
        v1 = np.zeros(256)
        v2 = np.zeros(256)
        for i in source_body:
            v1+=model.wv[i]
        v1 /= len(source_body)

        for i in target_body:
            v2+=model.wv[i]
        v2 /= len(target_body)
        if cos_sim(v1,v2):
            word2vec_sim_list.append(cos_sim(v1,v2))
        else:word2vec_sim_list.append(0)
    print(len(word2vec_sim_list))
    return word2vec_sim_list

def training_Doc2vec(corpus):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
    model = Doc2Vec(vector_size=256, alpha=0.025, min_alpha=0.00025, min_count=1, dm=1)
    model.build_vocab(tagged_data)
    for epoch in range(20):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("./model/doc2vec_256.model")

def Doc2vec_sim():
    doc2vec_sim_list = []
    model = Doc2Vec.load("./model/doc2vec_256.model")
    for index in range(len(body_text)):
        source_body=body_text['source_body'].loc[index]
        target_body=body_text['target_body'].loc[index]

        source_body = word_tokenize(source_body)
        target_body= word_tokenize(target_body)

        v1=model.infer_vector(source_body)
        v2=model.infer_vector(target_body)

        if cos_sim(v1, v2):
            doc2vec_sim_list.append(cos_sim(v1, v2))
        else:
            doc2vec_sim_list.append(0)

    return doc2vec_sim_list

def construct_neg_dataset():

    body_text = data_processing()  #
    #### 构造neg数据集
    df_neg= pd.read_csv('./Data/vue_ilinker_neg.csv',encoding='utf-8')
    for index in range(len(body_text)):

        random_list=[]

        for i in range(len(body_text)):

            if body_text['source_body'].loc[index]!=body_text['source_body'].loc[i]:
                random_list.append(i)

        j=random.sample(random_list,1)[0]

        df_neg['source_repo'].loc[index]= body_text['source_repo'].loc[index]
        df_neg['source_num'].loc[index] = body_text['source_num'].loc[index]
        df_neg['source_title'].loc[index] = body_text['source_title'].loc[index]
        df_neg['source_body'].loc[index] = body_text['source_body'].loc[index]
        df_neg['target_repo'].loc[index] = body_text['target_repo'].loc[j]
        df_neg['target_num'].loc[index] = body_text['target_num'].loc[j]
        df_neg['target_title'].loc[index] = body_text['target_title'].loc[j]
        df_neg['target_body'].loc[index] = body_text['target_body'].loc[j]

    df_neg.to_csv('./Data/vue_ilinker_neg.csv', index=False)



# find_body()
# construct_neg_dataset()
# body_text=data_processing()
# # body_text=data_processing_2()
# # print(body_text)
# corpus=[]
# corpus.extend(body_text['source_body'])
# corpus.extend(body_text['target_body'])
#
# ##### tf_idf 计算
#
# tf_idf_similarity(body_text,corpus)
# # body_text['tf_idf_sim']=tf_idf_similarity(body_text,corpus)
# body_text['tf_idf_sim_neg']=tf_idf_similarity(body_text,corpus)
# #
# # ###### word2vec 计算
# training_word2vec(corpus)
# # body_text['w2v_sim']=word2vec_sim()
# # # body_text.to_csv('./Data/vue_ilinker.csv', index=False)
# # # training_word2vec(corpus)
# body_text['w2v_sim_neg']=word2vec_sim()
# # # body_text.to_csv('./react/react_ilinker_neg.csv', index=False)
# # # #
# # # # #### doc2vec 计算
# training_Doc2vec(corpus)
# # body_text['d2v_sim']=Doc2vec_sim()
# body_text['d2v_sim_neg']=Doc2vec_sim()
# # body_text.to_csv('./Data/vue_ilinker.csv', index=False)
# body_text.to_csv('./react/vue_ilinker_neg.csv', index=False)

##### 总体 sim计算
# all_sim_list=[]
# df_1=pd.read_csv('./react/react_ilinker.csv', encoding='utf-8')
# for index in range(len(df_1)):
#     all_sim=(df_1['tf_idf_sim'].loc[index]+df_1['w2v_sim'].loc[index]+df_1['d2v_sim'].loc[index])/3
#     all_sim_list.append(all_sim)
# all_sim_list=[x for x in all_sim_list if x]
#
# all_neg_list=[]
# df_2=pd.read_csv('./react/react_ilinker_neg.csv', encoding='utf-8')
# for index in range(len(df_2)):
#     all_sim=(df_2['tf_idf_sim_neg'].loc[index]+df_2['w2v_sim_neg'].loc[index]+df_2['d2v_sim_neg'].loc[index])/3
#     all_neg_list.append(all_sim)
# all_neg_list=[x for x in all_sim_list if x]

###### 计算指标

def recent_metrics(recent_index_path):

    recent_test_set= pd.read_csv(recent_index_path, sep='\t', header=None, names=['issue_id', 'issue', 'pr_id','pr'],
                           keep_default_na=False, encoding='utf-8')

    df_1 = pd.read_csv('./Data/vue_ilinker.csv', encoding='utf-8')
    all_sim_list = []
    for index in range(len(recent_test_set)):

        tf_idf_sim=df_1[(df_1['source_repo'].replace(' ','/')==recent_test_set['issue'].loc[index].split('__')[0])&(df_1['source_num']==int(recent_test_set['issue'].loc[index].split('__')[1]))&
                        (df_1['target_repo'].replace(' ','/')==recent_test_set['pr'].loc[index].split('__')[0])&(df_1['target_num']==int(recent_test_set['pr'].loc[index].split('__')[1]))]['tf_idf_sim']


        w2v_sim = df_1[(df_1['source_repo'].replace(' ','/') == recent_test_set['issue'].loc[index].split('__')[0]) & (
                    df_1['source_num'] == int(recent_test_set['issue'].loc[index].split('__')[1])) &
                          (df_1['target_repo'].replace(' ','/') == recent_test_set['pr'].loc[index].split('__')[0]) & (
                                      df_1['target_num'] == int(recent_test_set['pr'].loc[index].split('__')[1]))]['w2v_sim']

        d2v_sim = df_1[(df_1['source_repo'].replace(' ','/') == recent_test_set['issue'].loc[index].split('__')[0]) & (
                    df_1['source_num'] == int(recent_test_set['issue'].loc[index].split('__')[1])) &
                          (df_1['target_repo'].replace(' ','/') == recent_test_set['pr'].loc[index].split('__')[0]) & (
                                      df_1['target_num'] == int(recent_test_set['pr'].loc[index].split('__')[1]))]['d2v_sim']

        if len(list(tf_idf_sim))!=0 and len(list(w2v_sim))!=0 and len(list(d2v_sim))!=0:
            all_sim=list(tf_idf_sim)[0]+list(w2v_sim)[0]+list(d2v_sim)[0]
            all_sim_list.append(all_sim/3)
        else:
            all_sim_list.append(0)
    print(len(all_sim_list))

    for index in range(len(all_sim_list)):
        if all_sim_list[index]>0:
            all_sim_list[index]=all_sim_list[index]
        else:
            all_sim_list[index]=random.uniform(0, 1)

    df_2 = pd.read_csv('./Data/vue_ilinker_neg.csv', encoding='utf-8')

    for index in range(len(recent_test_set)):

        tf_idf_sim=df_2[(df_2['source_repo'].replace(' ','/')==recent_test_set['issue'].loc[index].split('__')[0])&(df_2['source_num']==int(recent_test_set['issue'].loc[index].split('__')[1]))]['tf_idf_sim_neg']

        w2v_sim = df_2[(df_2['source_repo'].replace(' ','/') == recent_test_set['issue'].loc[index].split('__')[0]) & (
                    df_2['source_num'] == int(recent_test_set['issue'].loc[index].split('__')[1]))]['w2v_sim_neg']

        d2v_sim = df_2[(df_2['source_repo'].replace(' ','/') == recent_test_set['issue'].loc[index].split('__')[0]) & (
                    df_2['source_num'] == int(recent_test_set['issue'].loc[index].split('__')[1]))]['d2v_sim_neg']

        if len(list(tf_idf_sim))!=0 and len(list(w2v_sim))!=0 and len(list(d2v_sim))!=0:
            all_sim=list(tf_idf_sim)[0]+list(w2v_sim)[0]+list(d2v_sim)[0]
            all_sim_list.append(all_sim/3)
        else:
            all_sim_list.append(0)

    for index in range(len(all_sim_list)):
        if all_sim_list[index]>0:
            all_sim_list[index]=all_sim_list[index]
        else:
            all_sim_list[index]=random.uniform(0, 1)

    for index in range(len(all_sim_list)):
        if all_sim_list[index]>0.5:
            all_sim_list[index]=1
        else:
            all_sim_list[index] = 0
    print(len(all_sim_list))
    # np.save(r"./Data/ilinker_results.npy", np.array(all_sim_list).astype(int))

recent_metrics(recent_index_path='./Data\index/recent_index.txt')
