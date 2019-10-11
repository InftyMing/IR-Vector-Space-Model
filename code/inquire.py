import clean_text
import pickle
from tkinter import _flatten
import numpy as np
import collections

# 读取pickle文件中的序列化对象
def read_pkl(tarfile):
    tf = open(tarfile, 'rb')
    _list = pickle.load(tf)
    tf.close()
    return _list

#将矩阵x每列归一化（normalized）
def normalizeCols(x):
    x_norm = np.linalg.norm(x, axis=0, keepdims=True)
    x_normalized = x / x_norm
    return x_normalized

# 对用户的文本进行处理，统计词频
def clean_query_text(query_text):
    query_text = clean_text.clean_text(query_text)
    statistics = collections.Counter(query_text).most_common() # 统计词频
    for unit in statistics:
        statistics[statistics.index(unit)] = list(unit)
    return statistics

# 向量空间模型查询
def Vector_Space_Model(query_text, doc_count, query_type):
    if query_type == 1:
        a = []
        a.append(query_text)
        query_text = a
        index_list = read_pkl('author.pkl')
    if query_type == 2:
        query_text_2 = clean_query_text(query_text)
        query_text = [item[0] for item in query_text_2]
        index_list = read_pkl('subject.pkl')
    if query_type == 3:
        query_text_2 = clean_query_text(query_text)
        query_text = [item[0] for item in query_text_2]
        index_list = read_pkl('body.pkl')
    word_dic = [item[0] for item in index_list] # 词的列表
    match_list = []
    for word in query_text:
        if word in word_dic:
            match_list.append(index_list[word_dic.index(word)])
        if word not in word_dic:
            continue
    if len(match_list) == 0:
        print('Found nothing! Please enter a more accurate query-statement!')
        return None
    # 提取列表中的df_t和tf_td
    df_t = np.array([item[1] for item in match_list]).reshape(-1, 1) # 出现t的文档数目
    tf_td = np.array([item[2] for item in match_list]).reshape(-1, 1) # t在文档d中出现的次数
    # 对用户查询进行tf_idf权重计算并进行归一化处理
    weight = (1 + np.log10(tf_td)) * np.log10(doc_count / df_t)
    query_normalize = normalizeCols(weight) # 得到一维列向量，查询的归一化权重
    # 提取列表中的文档索引和tf-idf权重
    files_weight = [[item[i] for i in range(3, len(item))] for item in match_list]
    # 将新的match_word中的词提取成一个一维列表
    matched_words = [item[0] for item in match_list]
    # 将所有在倒排索引中出现词的文件进行整合，之后去重并排序，目的是构建矩阵
    all_files_index = [[item[i] for i in range(0, int(len(item) / 2))] for item in files_weight]
    all_index_list = list(set(list(_flatten(all_files_index))))
    all_index_list.sort()   # 结果为一个升序排列的一维数组
    # 初始化矩阵
    index_weight = np.zeros((len(matched_words), len(all_index_list)))
    tf_idf = [[item[i] for i in range(int(len(item) / 2), len(item))] for item in files_weight]
    # 权重矩阵构建完成
    for line in all_files_index:
        for i in range(0, len(line)):
            index_weight[all_files_index.index(line)][all_index_list.index(line[i])] = tf_idf[all_files_index.index(line)][i]
    index_weight = normalizeCols(index_weight)  # 归一化
    # 创建余弦相似度二维矩阵，矩阵每一行第一列为文件索引，最后一列为余弦相似度
    cosine_score = np.zeros((len(all_index_list), 2))
    for i in range(0, len(all_index_list)):
        cosine_score[i][0] = all_index_list[i]
        cosine_score[i][1] = np.sum(index_weight[:, i] * query_normalize)
    cosine_score = cosine_score[np.lexsort(-cosine_score.T)]
    return cosine_score

def find_doc_path(cosine_score):
    index_doc = read_pkl('doc_index.pkl')
    _index = [line[0] for line in index_doc]
    doc_path = []
    for item in cosine_score:
        doc_path.append(index_doc[_index.index(item[0])][1])
    print('According to your query statement, we have found the following %d e-mails.' % len(cosine_score))
    print('(Here are the path of these e-mail, ranking as the degree of correlation from high to low.)')
    for item in doc_path:
        print(item)
    return doc_path

#print(read_pkl('author.pkl'))
#print(read_pkl('subject.pkl'))
#print(read_pkl('body.pkl'))