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


# load TFIDF模型
def loadModelTFIDF(mdlFile, dicFile, idxFile):
    # 载入字典
    dictionary = corpora.Dictionary.load(dicFile)
    # 载入TFIDF模型和索引
    tfidfModel = models.TfidfModel.load(mdlFile)
    indexTfidf = similarities.MatrixSimilarity.load(idxFile)

    return (tfidfModel, dictionary, indexTfidf)


# 利用TFIDF模型计算相似度
def simTFIDF(test_data, tfidfModel, dictionary, indexTfidf):
    # 处理测试数据
    query_bow = dictionary.doc2bow(test_data)

    # 使用TFIDF模型向量化
    tfidfvect = tfidfModel[query_bow]

    print tfidfvect

    # TFIDF相似性
    simstfidf = indexTfidf[tfidfvect]

    return simstfidf


# load LDA模型
def loadModelLDA(ldaMDL, dicFile, idxFile):
    # 载入字典
    dictionary = corpora.Dictionary.load(dicFile)
    # 载入TFIDF模型
    #     tfidfModel = models.TfidfModel.load(tfidfMDL)
    # 载入LDA模型和索引
    ldaModel = models.LdaModel.load(ldaMDL)
    indexLDA = similarities.MatrixSimilarity.load(idxFile)

    return (ldaModel, dictionary, indexLDA)


# 利用LDA模型计算相似度
def simLDA(test_data, tfidfModel, dictionary, ldaModel, indexLDA):
    # 处理测试数据
    query_bow = dictionary.doc2bow(test_data)
    # 使用TFIDF模型向量化
    tfidfvect = tfidfModel[query_bow]
    # 然后LDA向量化,因为我们训练时的LDA是在TFIDF基础上做的,所以用itidfvect再向量化一次
    ldavec = ldaModel[tfidfvect]
    # LDA相似性
    simlda = indexLDA[ldavec]
    return simlda


# 获取相似文章的题目
def getTitleIdx(simValue, TitleIdx):
    # 标题索引
    title_set = []
    result = []
    # 读入训练数据
    f = open(TitleIdx)
    lines = f.readlines()
    for line in lines:
        title_set.append(line.replace('\n', ''))
    f.close()

    for item in simValue:
        idx, value = item
        print "========================"
        print "similarity article :%s,simvalue= %s" % (title_set[idx].split("\t")[0], value)
        print "article content : ", title_set[idx].split("\t")[1]
        print "========================"


# 计算相似度
def getSimTFIDF(model_info, inFile, TitleIdx):
    title_set, test_set = getTrainSet(inFile)
    mdl, dic, idx = model_info
    # load module
    tfidfModel, dict_tfidf, indexTfidf = loadModelTFIDF(mdl, dic, idx)

    for i, text in enumerate(test_set):
        # 返回相似度最高的前5篇文章
        # TFIDF 模型
        simValue = sorted(enumerate(simTFIDF(text, tfidfModel, dict_tfidf, indexTfidf)), key=lambda item: -item[1])[:5]
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@"
        print "Test Article = ", title_set[i]
        print "TFIDF Similarity =%s" % (simValue)
        getTitleIdx(simValue, TitleIdx)
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@"


# 计算相似度
def getSimLDA(model_info, inFile, TitleIdx):
    title_set, test_set = getTrainSet(inFile)
    tfidf_info, lda_info = model_info
    tfidf_mdl, tfidf_dic, tfidf_idx = tfidf_info
    LDA_mdl, LDA_dic, LDA_idx = lda_info
    # load module
    tfidfModel, dict_tfidf, indexTfidf = loadModelTFIDF(tfidf_mdl, tfidf_dic, tfidf_idx)
    # load module
    ldaModel, dict_lda, indexLDA = loadModelLDA(LDA_mdl, LDA_dic, LDA_idx)
    for i, text in enumerate(test_set):
        # 返回相似度最高的前5篇文章
        # LDA 模型
        simValue = sorted(enumerate(simLDA(text, tfidfModel, dict_lda, ldaModel, indexLDA)),
                          key=lambda item: -item[1])[:5]
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@"
        print "Test Article = ", title_set[i]
        print "LDA Similarity =%s" % (simValue)
        getTitleIdx(simValue, TitleIdx)
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@"


def main():
    tfidf_info = ("./model/all_test_TFIDF.mdl",
                  "./model/all_test_TFIDF.dic",
                  "./model/all_test_TFIDF.idx")
    lda_info = ("./model/all_test_LDA50TOPIC.mdl",
                "./model/all_test_LDA50TOPIC.dic",
                "./model/all_test_LDA50TOPIC.idx")
    inFile = "./data/test.txt"
    TitleIdx = "./data/processed/all_merge.txt"
    # 利用TFIDF 计算相似度
    getSimTFIDF(tfidf_info, inFile, TitleIdx)
    # 利用LDA 计算相似度
    getSimLDA((tfidf_info, lda_info), inFile, TitleIdx)


if __name__ == '__main__':
    main()


