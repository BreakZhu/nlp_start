# -*— encoding: utf-8 -*-
import os
import jieba

"""
生成原始语料文件夹下文件列表
"""
data_simple = 'E:\\sougo\\souhusimple\\'
stop_file = 'E:\\sougo\\stopwords'


def listdir(simple_path, list_name):
    for fl in os.listdir(simple_path):
        file_path = os.path.join(simple_path, fl)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


# 获取停用词表
def get_stop_words():
    stop_fl = open(stop_file, 'rb').read().decode('utf-8').split('\r\n')
    return set(stop_fl)


# 去除内容词中的停用词
def rm_stop_words(conten_words):
    conten_words = list(conten_words)
    stop_words = get_stop_words()
    # 这个很重要，注意每次pop之后总长度是变化的
    for i in range(len(conten_words))[::-1]:
        if conten_words[i] in stop_words:
            conten_words.pop(i)
        elif conten_words[i].isdigit():
            conten_words.pop(i)
    return conten_words


# 如果统计词频的时候一个单词出现的次数过少，也不用统计这个词
def rm_word_freq_so_little(dictionary, freq_thred):
    small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < freq_thred]
    dictionary.filter_tokens(small_freq_ids)
    dictionary.compactify()


def word_cut_jieba(content):
    return rm_stop_words(list(jieba.cut(content, cut_all=False)))
