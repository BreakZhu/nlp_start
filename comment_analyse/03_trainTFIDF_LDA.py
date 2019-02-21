# -*- encoding: utf-8 -*-

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

from gensim import corpora, models, similarities, utils

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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


# 把筛选以后的结果写入txt文件
def writeTitleSet(fileName, data, mode):
    fw = open(fileName, mode)
    for w in data:
        fw.writelines(w + "\n")
    fw.close()


# 训练模型
def trainTFIDF(train_set, mdlFile, dicFile, idxFile, title_set, TitleFile):
    # 生成字典
    dictionary = corpora.Dictionary(train_set)
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=None)
    dictionary.save(dicFile)

    # 生成语料
    corpus = [dictionary.doc2bow(text) for text in train_set]

    # 使用数字语料生成TFIDF模型
    tfidfModel = models.TfidfModel(corpus)
    # 存储tfidfModel
    tfidfModel.save(mdlFile)

    # 把全部语料向量化成TFIDF模式,这个tfidfModel可以传入二维数组
    tfidfVectors = tfidfModel[corpus]
    # 建立索引并保存
    indexTfidf = similarities.MatrixSimilarity(tfidfVectors)
    indexTfidf.save(idxFile)
    # 把文件标题索引写入文件
    writeTitleSet(TitleFile, title_set, "a")


# 训练模型
def trainLDA(train_set, mdlFile, dicFile, idxFile, title_set, TitleFile):
    # 生成字典
    dictionary = corpora.Dictionary(train_set)
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=None)
    dictionary.save(dicFile)
    # 生成语料
    corpus = [dictionary.doc2bow(text) for text in train_set]

    # 使用数字语料生成TFIDF模型
    tfidfModel = models.TfidfModel(corpus)

    # 把全部语料向量化成TFIDF模式,这个tfidfModel可以传入二维数组
    tfidfVectors = tfidfModel[corpus]

    # 通过TFIDF向量生成LDA模型,id2word表示编号的对应词典,num_topics表示主题数,我们这里设定的10
    lda = models.LdaModel(tfidfVectors, id2word=dictionary, num_topics=50)
    # 把模型保存下来
    lda.save(mdlFile)
    # 把所有TFIDF向量变成LDA的向量
    corpus_lda = lda[tfidfVectors]
    # 建立索引,把LDA数据保存下来
    indexLDA = similarities.MatrixSimilarity(corpus_lda)
    indexLDA.save(idxFile)
    # 把文件标题索引写入文件
    writeTitleSet(TitleFile, title_set, "w")


def trainModel():
    inFile = "./data/processed/all_summary.txt"
    title_set, train_set = getTrainSet(inFile)
    TFIDF_mdl = "./model/all_test_TFIDF.mdl"
    TFIDF_dic = "./model/all_test_TFIDF.dic"
    TFIDF_idx = "./model/all_test_TFIDF.idx"
    LDA_mdl = "./model/all_test_LDA50TOPIC.mdl"
    LDA_dic = "./model/all_test_LDA50TOPIC.dic"
    LDA_idx = "./model/all_test_LDA50TOPIC.idx"
    TitleFile = "./model/infoSet.txt"
    # 训练TFIDF
    trainTFIDF(train_set, TFIDF_mdl, TFIDF_dic, TFIDF_idx, title_set, TitleFile)
    # 训练LDA
    trainLDA(train_set, LDA_mdl, LDA_dic, LDA_idx, title_set, TitleFile)


def main():
    trainModel()


if __name__ == '__main__':
    main()
