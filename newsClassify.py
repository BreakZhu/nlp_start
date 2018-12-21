# -*-coding:utf-8 -*-

"""
本文档负责实际读取语料库文件
训练LR模型
过程中保存词典、语料和训练后的模型
"""
import numpy as np
from sklearn.linear_model.logistic import  *
from gensim import corpora, models, similarities
import jieba
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
from utils.wordUtils import rm_stop_words, rm_word_freq_so_little, listdir

data_simple = 'E:\\sougo\\souhusimple\\'

if __name__ == '__main__':
    freq_thred = 10        # 当一个单词在所有语料中出现次数小于这个阈值，那么该词语不应被计入词典中
    # 字典
    dictionary = corpora.Dictionary()
    # 词袋
    bow = []

    labels_count = []
    list_name = []
    listdir(data_simple, list_name)
    count = 0

    for file_path in list_name[0:2]:
        print(file_path)
        fl = open(file_path, 'rb').read().decode('utf-8').split('\n')
        class_count = 0
        for text in fl:
            # 打标签
            class_count = class_count + 1
            content = text
            # 分词
            word_list = list(jieba.cut(content, cut_all=False))
            # 去停用词
            word_list = rm_stop_words(word_list)

            dictionary.add_documents([word_list])
            """
            转化成词袋gensim包中的dic实际相当于一个map doc2bow方法，对没有出现过的词语，在dic中增加该词语
            如果dic中有该词语，则将该词语序号放到当前word_bow中并且统计该序号单词在该文本中出现了几次
            注意这里保存的是稀疏矩阵
            单个词向量：( 5 , 2 ) 5是该单词在dictionary中的序号为5，2是在这篇文章中出现了两次。
            一篇文章矩阵： [ (5 ,2) , (3 , 1) ] 在该文章中出现了5号单词两次，3号单词1次。
            """
            word_bow = dictionary.doc2bow(word_list)
            bow.append(word_bow)
        labels_count.append(class_count-1)
    # with open('dictionary.pkl', 'wb') as f1:
    #   pickle.dump(dictionary, f1)

    # 去除过少单词 ps:可能导致维数不同
    rm_word_freq_so_little(dictionary, freq_thred)
    dictionary.save('dicsave.dict')
    corpora.MmCorpus.serialize('bowsave.mm', bow)

    tfidf_model = models.TfidfModel(corpus=bow, dictionary=dictionary)

    # with open('tfidf_model.pkl', 'wb') as f2:
    #     pickle.dump(tfidf_model, f2)
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
    print(line_count)
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
    for i in range(labels_count[0]):
        labels[i] = 1

    """
    分割训练集和测试集
    """
    rarray=np.random.random(size=line_count)
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(line_count-1):
        if rarray[i]<0.8:
            x_train.append(tfidf_matrix[i,:])
            y_train.append(labels[i])
        else:
            x_test.append(tfidf_matrix[i,:])
            y_test.append(labels[i])

    # x_train,x_test,y_train,y_test = train_test_split(tfidf_matrix,labels,test_size=0.3,random_state=0)

    """
    LR模型分类训练
    """
    classifier = LogisticRegression()

    classifier.fit(x_train, y_train)
    #
    # with open('LR_model.pkl', 'wb') as f:
    #     pickle.dump(classifier, f)

    print(classification_report(y_test,classifier.predict(x_test)))