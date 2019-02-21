# -*-coding:utf-8 -*-

"""
本文档负责实际读取语料库文件
训练LR模型
过程中保存词典、语料和训练后的模型
"""
import numpy as np
import time
from sklearn import *
from gensim import corpora, models, similarities
import jieba
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.word_opt_util import rm_stop_words, rm_word_freq_so_little, listdir

data_simple = 'E:\\sougo\\souhusimple\\'
TimeTag = time.strftime('%Y%m%d', time.localtime(time.time()))

if __name__ == '__main__':
    freq_thred = 10  # 当一个单词在所有语料中出现次数小于这个阈值，那么该词语不应被计入词典中
    # 字典
    dictionary = corpora.Dictionary()
    # 词袋
    bow = []
    labels_name = []
    labels_count = []
    list_name = []
    listdir(data_simple, list_name)
    count = 0

    #  for file_path in list_name[0:2]:
    for file_path in list_name[0:3]:
        file_name = file_path.split("\\")[-1].split(".")[0]
        labels_name.append(file_name)
        print file_path
        fl = open(file_path, 'rb')
        class_count = 0
        for text in fl:
            # 打标签
            class_count = class_count + 1
            content = text
            # 分词
            # 本文处理的语料均为中文，因此处理中文分词需要使用结巴分词。结巴分词的原理大致是：
            # 先使用正则表达式粗略划分，然后基于trie树高速扫描，将每个句子构造有向无环图，使用动态规划查找最大概率路径，
            # 基于词频寻找最佳切分方案，最后对于未登录的单词（词表里没有的词语），采用HMM模型，维特比算法划分
            word_list = list(jieba.cut(content, cut_all=False))
            # 去停用词
            # 中文中很多词语虽然在文章中大量出现，但对文章分类并没有什么实际意义。
            # 比如“只”、“的”、“应该”这样的词语，对它们的计算既浪费空间时间也可能影响最终分类结果。
            # 因此需要先建立一个词表，将样本语料分词后出现在该词表中的单词去掉
            word_list = rm_stop_words(word_list)

            # 这里用gensim提供的 dictionary ，它相当于一个map，每个单词如果出现在里面，
            # 那么就会在当前文章向量中记录该单词在词典中的序号和在该文章中的频率。
            # 生成词向量仅需要调用dictionary.doc2bow()方法即可生成。注意这里保存的是稀疏矩阵。具体格式为：
            # 单个词向量：( 5 , 2 )
            # 5是该单词在dictionary中的序号为5，2是在这篇文章中出现了两次。
            # 一篇文章矩阵： [ (5 ,2) , (3 , 1) ]
            # 在该文章中出现了5号单词两次，3号单词1次。
            dictionary.add_documents([word_list])

            # 转化成词袋gensim包中的dic实际相当于一个map doc2bow方法，对没有出现过的词语，在dic中增加该词语
            # 如果dic中有该词语，则将该词语序号放到当前word_bow中并且统计该序号单词在该文本中出现了几次
            # 注意这里保存的是稀疏矩阵
            # 单个词向量：( 5 , 2 ) 5是该单词在dictionary中的序号为5，2是在这篇文章中出现了两次。
            # 一篇文章矩阵： [ (5 ,2) , (3 , 1) ] 在该文章中出现了5号单词两次，3号单词1次。
            word_bow = dictionary.doc2bow(word_list)
            bow.append(word_bow)
            if class_count == 1000:
                break
        labels_count.append(class_count)

    # with open('dictionary.pkl', 'wb') as f1:
    #   pickle.dump(dictionary, f1)

    # 去除过少单词 ps:可能导致维数不同
    """
    如果统计词频的时候一个单词出现的次数过少，也不用统计这个词
    """
    rm_word_freq_so_little(dictionary, freq_thred)

    dictionary.save('model/dicsave.dict_{}'.format(TimeTag))
    # 序列化到硬盘
    corpora.MmCorpus.serialize('model/bowsave.mm_{}'.format(TimeTag), bow)

    # dictionary = dictionary.load('model/dicsave.dict_{}'.format(TimeTag))
    # 从硬盘加载
    # bow = corpora.MmCorpus('model/bowsave.mm_{}'.format(TimeTag))

    tfidf_model = models.TfidfModel(corpus=bow, dictionary=dictionary)
    with open('./model/tfidf_model_{}.pkl'.format(TimeTag), 'wb') as f2:
        pickle.dump(tfidf_model, f2)
    """
    训练tf-idf模型
    """
    corpus_tfidf = [tfidf_model[doc] for doc in bow]
    """
    将gensim格式稀疏矩阵转换成可以输入scikit-learn模型格式矩阵
    """
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus_tfidf:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    print line_count
    """
    csr_matrix
    csr_matrix(Compressed Sparse Row matrix)或csc_matric(Compressed Sparse Column marix)，为压缩稀疏矩阵的存储方式
    scipy.sparse.csr_matrix
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])
    上述方式为按照row行来压缩 
    （1）data表示数据，为[1, 2, 3, 4, 5, 6] 
    （2）shape表示矩阵的形状 
    （3）indices表示对应data中的数据，在压缩后矩阵中各行的下标，如：数据1在某行的0位置处，数据2在某行的2位置处，
    数据6在某行的2位置处。 
    （4）indptr表示压缩后矩阵中每一行所拥有数据的个数，如：[0 2 3 6]表示从第0行开始数据的个数，
        0表示默认起始点，0之后有几个数字就表示有几行，第一个数字2表示第一行有2 - 0 = 2个数字，因而数字1，2都第0行，
        第二行有3 - 2 = 1个数字，因而数字3在第1行，以此类推。
    """
    tfidf_matrix = csr_matrix((data, (rows, cols))).toarray()
    count = 0
    for ele in tfidf_matrix:
        # print(ele)
        # print(count)
        count = count + 1

    # cut label 1 mil label 0
    """
    生成labels
    """
    labels = np.zeros(sum(labels_count) + 1)
    index = 0
    for num, val in enumerate(labels_count):
        for i in range(labels_count[num]):
            labels[index] = num
            index += 1

    """
    分割训练集和测试集
    """
    rarray = np.random.random(size=line_count)
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(line_count - 1):
        if rarray[i] < 0.6:
            x_train.append(tfidf_matrix[i, :])
            y_train.append(labels[i])
        else:
            x_test.append(tfidf_matrix[i, :])
            y_test.append(labels[i])

    print set(y_train)
    print set(y_test)
    # x_train,x_test,y_train,y_test = train_test_split(tfidf_matrix,labels,test_size=0.3,random_state=0)

    """
    LR模型分类训练
    """
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    with open('./model/LR_model_{}.pkl'.format(TimeTag), 'wb') as f:
        pickle.dump(classifier, f)

    print classification_report(y_test, classifier.predict(x_test))
