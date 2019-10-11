import email_handling as eh
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import clean_text as ct
import collections
import numpy as np
import math
import pickle

# 将序列化对象存储在pickle文件中
def write_pkl(outputfile, item):
    of = open(outputfile, 'wb')
    pickle.dump(item, of)
    of.close()
    print("finish writing in %s !" % outputfile)

# 把内含元组的列表转换成二维列表
def tuple_to_list(_list):   
    for unit in _list:
        _list[_list.index(unit)] = list(unit)
    return _list

# 对邮件作者构建倒排索引，输入索引加文件名的二维列表
def create_author_InvertedIndex(index_file_list, header):
    file_set = index_file_list
    inverted_index = []
    for i in range(0, len(file_set)):
        cleaned = eh.extract_header(eh.read_email(file_set[i][1]), header)
        statistics = collections.Counter(cleaned).most_common() # 统计词频
        sub_list = tuple_to_list(statistics)
        for j in range(0, len(sub_list)):   # 向所有词后添加文件索引
            sub_list[j].append(file_set[i][0])
        if len(inverted_index) == 0:
            inverted_index += sub_list
        else:
            combine_list(inverted_index, sub_list)
    dict_sort = sort_as_dict(inverted_index)
    return re_comb(dict_sort, len(file_set))

# 对邮件标题构建倒排索引，输入索引加文件名的二维列表
def create_subject_InvertedIndex(index_file_list, header):
    file_set = index_file_list
    inverted_index = []
    for i in range(0, len(file_set)):
        cleaned = get_cleaned_subject(file_set[i][1], header)
        if len(cleaned) != 0:
            statistics = collections.Counter(cleaned).most_common() # 统计词频
            sub_list = tuple_to_list(statistics)
            for j in range(0, len(sub_list)):   # 向所有词后添加文件索引
                sub_list[j].append(file_set[i][0])
            if len(inverted_index) == 0:
                inverted_index += sub_list
            else:
                combine_list(inverted_index, sub_list)
    dict_sort = sort_as_dict(inverted_index)
    return re_comb(dict_sort, len(file_set))

# 对邮件内容构建倒排索引，输入索引加文件名的二维列表
def create_body_InvertedIndex(index_file_list):
    file_set = index_file_list
    inverted_index = []
    for i in range(0, len(file_set)):
        cleaned = get_cleaned_body(file_set[i][1])
        statistics = collections.Counter(cleaned).most_common() # 统计词频
        sub_list = tuple_to_list(statistics)
        for j in range(0, len(sub_list)):   # 向所有词后添加文件索引
            sub_list[j].append(file_set[i][0])
        if len(inverted_index) == 0:
            inverted_index += sub_list
        else:
            combine_list(inverted_index, sub_list)
    dict_sort = sort_as_dict(inverted_index)
    return re_comb(dict_sort, len(file_set))

# 输入两个二维列表
def combine_list(list_1, list_2):   
    word_1 = [item[0] for item in list_1]
    word_2 = [item[0] for item in list_2]
    for i in range(0, len(word_2)):
        if word_2[i] in word_1:
            _index = word_1.index(word_2[i])
            #list_1[_index][1] += list_2[i][1]
            list_1[_index].append(list_2[i][1])
            list_1[_index].append(list_2[i][2])
        else:
            w = np.array(list_2[i]).reshape(1, -1).tolist()
            w[0][1] = int(w[0][1])
            w[0][2] = int(w[0][2])
            list_1 += w
    return list_1

# 输入构建好的单词加索引的二维列表
def sort_as_dict(_list):    
    l = [item[0] for item in _list]
    word = sorted(l)
    dict = []
    for i in range(0, len(word)):
        if word[i] in l:
            dict.append(_list[l.index(word[i])])
    return dict

def get_cleaned_subject(tarpath, header):
    email = eh.read_email(tarpath)
    subject = eh.extract_header(email, header)
    if len(subject) != 0:
        cleaned = ct.clean_text(subject[0])
        return cleaned
    if len(subject) == 0:
        return []

def get_cleaned_body(tarpath):
    email = eh.read_email(tarpath)
    body = eh.extract_body(email)
    cleaned = ct.clean_text(body)
    return cleaned

# tf_td：t在文档d中出现的次数
# df_t：出现t的文档数目
# 计算词的tf-idf权重

def re_comb(_list, sum_files):
    word = np.array([item[0] for item in _list]).reshape(-1,1).tolist()
    tf_td = [[-1 for col in range(1)] for raw in range(len(word))]
    files = [[-1 for col in range(1)] for raw in range(len(word))]
    for i in range(0, len(_list)):
        for j in range(0, int((len(_list[i])-1)/2)):
            if tf_td[i][0] == -1 or files[i][0] == -1:
                tf_td[i][0] = _list[i][2*j+1]
                files[i][0] = _list[i][2*j+2]
            else:
                tf_td[i].append(_list[i][2*j+1])
                files[i].append(_list[i][2*j+2])
        _sum = sum(tf_td[i])
        _len = len(files[i])
        tf_td[i].append(_sum)
        files[i].append(_len)
    word_sum = np.array([item.pop() for item in tf_td]).reshape(-1,1).tolist()
    doc_sum = np.array([item.pop() for item in files]).reshape(-1,1).tolist()
    N = sum_files
    for i in range(0, len(tf_td)):
        word[i].append(doc_sum[i][0])
        word[i].append(word_sum[i][0])
        for j in range(0, len(tf_td[i])):
            tf_td[i][j] = (1 + math.log10(tf_td[i][j])) * math.log10(N / doc_sum[i][0])
            word[i].append(files[i][j])
        for j in range(0, len(tf_td[i])):
            word[i].append(tf_td[i][j])
    return word

'''
对于倒排索引二维列表的每一行而言，每个数据依次表示的含义是：
文档中的词--该词出现文档总数--该词在所有文档中出现的总次数--|文档索引|--|tf_idf权重|
'''

