import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def remove_symbols(text):
    del_estr = string.punctuation + string.digits   # ascii中标点符号加数字
    replace = ' ' * len(del_estr)     # maketrans函数要求替换的长度相同
    tran_tab = str.maketrans(del_estr, replace)
    text = text.translate(tran_tab)
    return text

def clean_text(text):   # 传入字符串，输出只保留英文词干的列表
    text = text.lower()     # 转换成小写
    only_word = remove_symbols(text)     # 去除字符串中的标点和数字
    sub = nltk.word_tokenize(only_word)      # 利用nltk工具进行分词
    without_stopwords = [w for w in sub if not w in stopwords.words('english')] # 去除停用词
    stemmer = PorterStemmer()   # 提取词干
    cleaned_test = [stemmer.stem(s) for s in without_stopwords]
    return cleaned_test

