import codecs
import numpy as np
import os
from email.parser import Parser
import pickle

# 读取pickle文件中的序列化对象
def read_pkl(tarfile):
    tf = open(tarfile, 'rb')
    _list = pickle.load(tf)
    tf.close()
    return _list

# 将序列化对象存储在pickle文件中
def write_pkl(outputfile, item):
    of = open(outputfile, 'wb')
    pickle.dump(item, of)
    of.close()
    print("finish writing in %s !" % outputfile)

# 读取文件，每行以矩阵形式输出
def read_files(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        _list = f.readlines()
    for i in range(0, len(_list)):
        _list[i] = _list[i].rstrip('\n')
        _list[i] = _list[i].rstrip('\r')
    _line = np.array(_list)
    return _line

# 读取文件，作为一个整体的字符串输出
def read_email(path):
    if os.path.exists(path):
        with open(path, encoding='windows-1252') as f:
            print(path)
            email = f.read()
            return email
    else:
        return "file not exist!"

# 提取邮件头中想要的信息
def extract_header(email, header):  # email为经过read_mail函数读取出的字符串，header为想要读取信息的列表
    message = Parser().parsestr(email)
    _m = []
    for item in header:
        h = message[item]
        if h == None:   # 防止当h是None时，h的类型会变成NoneType，导致replace函数报错无法继续执行
            h = ''
        h.replace('\n','')  # 去掉字符串中的转义字符，下同
        h.replace('\t','') 
        if h != '':
            _m.append(h)
    return _m

# 提取邮件内容（正文主体），即去掉了文件头
def extract_body(email):    # email经过read_mail函数读取出的字符串
    message = Parser().parsestr(email)
    item = list(message)
    last_header = message[item[len(item) - 1]]
    where = email.find(last_header)
    body = email[where + len(last_header) + 1 : len(email)]
    return body

# 获取所有文件的目录以构建索引，以二维列表形式输出
def files_index(tarpath):
    _files = []
    for root,dirs,files in os.walk(tarpath):        
        for file in files:            #获取文件所属目录                       
            _files.append(os.path.join(root, file))
    #files = np.array(_files).reshape(-1,1)
    #index = np.arange(len(files)).reshape(-1,1)
    index = np.arange(len(_files)).tolist()
    #index_files = np.hstack((index, files))     # 纵向整合之后，所有的int类型变成了str类型
    index_files = [list(item) for item in zip(index, _files)]
    write_pkl('doc_num.pkl', len(index_files))
    write_pkl('doc_index.pkl', index_files)
    return index_files

def extract_index(matrix):
    # 这里的目的是再把索引单独提取出，转成矩阵形式，当然没啥用（重复性操作）
    _index = np.array([int(num) for num in list(matrix[:,0])]).reshape(-1,1)
    return _index
