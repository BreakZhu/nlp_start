# -*- encoding: utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import logging
from gensim import corpora, models, utils, matutils


# 获取训练数据
def getTrainSet(inFile):
    # 文章标题集
    title_set = []
    # 训练集
    train_set = []
    # 读入训练数据
    f = open(inFile)
    lines = f.readlines()
    for line in lines:
        article = line.replace('\n', '').split('\t')
        title = article[0]
        title_set.append(title)
        content = article[1:]
        train_set.append(content)
    f.close()
    return (title_set, train_set)


# 把文本内容转化为TFIDF向量
def vecTransformTFIDF(train_set):
    # 生成字典
    dictionary = corpora.Dictionary(train_set)
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=None)

    # 生成语料
    corpus = [dictionary.doc2bow(text) for text in train_set]
    # 使用数字语料生成TFIDF模型
    tfidfModel = models.TfidfModel(corpus)
    # 把全部语料向量化成TFIDF模式,这个tfidfModel可以传入二维数组
    corpus_vec = tfidfModel[corpus]
    # 把预料库转化为scipy稀疏矩阵
    matrix = matutils.corpus2csc(corpus_vec).toarray()
    print "*** Transform TFIDF Vector to matrix ***"
    print "matrix shape=", matrix.shape
    return matrix


# 把文本内容转化为LSI向量
def vecTransformLSI(train_set):
    # 生成字典
    dictionary = corpora.Dictionary(train_set)
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=None)
    # 生成语料
    corpus = [dictionary.doc2bow(text) for text in train_set]
    # 使用数字语料生成TFIDF模型
    tfidfModel = models.TfidfModel(corpus)
    # 把全部语料向量化成TFIDF模式,这个tfidfModel可以传入二维数组
    tfidfVectors = tfidfModel[corpus]
    # 通过TFIDF向量生成LSI模型,id2word表示编号的对应词典,num_topics表示主题数
    lsi = models.LsiModel(tfidfVectors, id2word=dictionary, num_topics=50)
    # 把所有TFIDF向量变成LSI的向量
    corpus_vec = lsi[tfidfVectors]
    # 把预料库转化为scipy稀疏矩阵
    matrix = matutils.corpus2csc(corpus_vec).toarray()
    print "*** Transform LSI Vector to matrix ***"
    print "matrix shape=", matrix.shape
    return matrix


# 把文本内容转化为LDA向量
def vecTransformLDA(train_set):
    # 生成字典
    dictionary = corpora.Dictionary(train_set)
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=None)

    # 生成语料
    corpus = [dictionary.doc2bow(text) for text in train_set]

    # 使用数字语料生成TFIDF模型
    tfidfModel = models.TfidfModel(corpus)

    # 把全部语料向量化成TFIDF模式,这个tfidfModel可以传入二维数组
    tfidfVectors = tfidfModel[corpus]

    # 通过TFIDF向量生成LDA模型,id2word表示编号的对应词典,num_topics表示主题数
    lda = models.LdaModel(tfidfVectors, id2word=dictionary, num_topics=50)

    # 把所有TFIDF向量变成LDA的向量
    corpus_vec = lda[tfidfVectors]

    # 把预料库转化为scipy稀疏矩阵
    matrix = matutils.corpus2csc(corpus_vec).toarray()
    print "*** Transform LDA Vector to matrix ***"
    print "matrix shape=", matrix.shape
    return matrix


def trainModel():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    inFile = "./data/processed/all_summary.txt"
    # 读入数据文件
    label, text = getTrainSet(inFile)
    # 转换数据，并写入文件
    vec = vecTransformTFIDF(text)
    np.save("./model/textData_TFIDF.npy", vec)
    # 转换数据，并写入文件
    #     vec = vecTransformLSI(text)
    #     np.save("./model/textData_LSI.npy",vec)
    # 转换数据，并写入文件
    #     vec = vecTransformLDA(text)
    #     np.save("./model/textData_LDA.npy",vec)


def main():
    trainModel()

if __name__ == '__main__':
    main()