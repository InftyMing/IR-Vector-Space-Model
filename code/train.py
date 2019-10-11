import email_handling as eh
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import clean_text as ct
import collections
import numpy as np
import create_InvertedIndex as ci
import codecs
import pickle

# 读取pickle文件中的序列化对象
def read_pkl(tarfile):
    tf = open(tarfile, 'rb')
    _list = pickle.load(tf)
    tf.close()
    return _list

def write_file(output_file, _list):
    with codecs.open(output_file,'w',encoding='utf-8') as g:
        for line in _list:
            line = str(line) + '\n'
            g.write(line)

# 对文档重新建立索引，原来矩阵，现在列表
tarpath = 'maildir/'
file_set = eh.files_index(tarpath)
'''
a = ['From']
s = ['Subject']
result_1 = ci.create_author_InvertedIndex(file_set, a)
write_file('author.txt', result_1)
ci.write_pkl('author.pkl', result_1)

result_2 = ci.create_subject_InvertedIndex(file_set, s)
write_file('subject.txt', result_2)
ci.write_pkl('subject.pkl', result_2)
'''
result_3 = ci.create_body_InvertedIndex(file_set)
write_file('body.txt', result_3)
ci.write_pkl('body.pkl', result_3)
# print("finish!")