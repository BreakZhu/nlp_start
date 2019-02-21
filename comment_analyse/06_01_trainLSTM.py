# -*- encoding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# import numpy as np
# from keras.layers.core import Activation, Dense
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
# from keras.preprocessing import sequence
# from sklearn.model_selection import train_test_split
# import collections  #用来统计词频
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
#
# # 获取训练数据
# def getTrainSet(inFile):
#     # 训练集
#     train_set = []
#     # 情感标签集
#     target_set = []
#     # 统计所有出现的词
#     word_ctr = collections.Counter()
#     # 评论的最大长度
#     maxlen = 0
#     len_ctr = collections.Counter()
#
#     # 读入训练数据
#     f = open(inFile)
#     lines = f.readlines()
#     for line in lines:
#         article = line.replace('\n', '').split('\t')
#
#         # 情感标签
#         target_set.append(article[1])
#         # 内容
#         content = article[2:]
#         train_set.append(content)
#
#         # 获得评论的最大长度
#         if len(content) > maxlen:
#             maxlen = len(content)
#
#         len_ctr[str(len(content))] += 1
#
#         # 统计所有出现的词
#         for w in content:
#             word_ctr[w] += 1
#
#     f.close()
#     print('max_len ', maxlen)
#     print('nb_words ', len(word_ctr))
#     print('mean lenth', len_ctr)
#     return (target_set, train_set, maxlen, word_ctr)
#
#
# # 把原始文本转化为由词汇表索引表示的矩阵
# def trainLSTM(inFile, outFile):
#     # 读入数据
#     target_set, data_set, maxlen, word_ctr = getTrainSet(inFile)
#
#     # 创建训练数据
#     X = np.empty(len(data_set), dtype=list)
#     y = np.array([int(i) for i in target_set])
#
#     #     print "X len = ",len(X)
#     #     print "y len = ",len(y)
#
#     # ('max_len ', 172)
#     # ('nb_words ', 5195)
#     MAX_FEATURES = 3500
#     MAX_SENTENCE_LENGTH = 70
#
#     # 对于不在词汇表里的单词，把它们用伪单词 UNK 代替。
#     # 根据句子的最大长度 (max_lens)，我们可以统一句子的长度，把短句用 0 填充。
#     # 接下来建立两个 lookup tables，分别是 word2index 和 index2word，用于单词和数字转换。
#     vocab_size = min(MAX_FEATURES, len(word_ctr)) + 2
#     word2index = {x[0]: i + 2 for i, x in enumerate(word_ctr.most_common(MAX_FEATURES))}
#     word2index["PAD"] = 0
#     word2index["UNK"] = 1
#     index2word = {v: k for k, v in word2index.items()}
#     # 对每一个文章做转换
#
#     i = 0
#     for news in data_set:
#         trs_news = []
#         for w in news:
#             if w in word2index:
#                 trs_news.append(word2index[w])
#             else:
#                 trs_news.append(word2index['UNK'])
#         X[i] = trs_news
#         i += 1
#
#     # 对文字序列做补齐 ，补齐长度=最长的文章长度 ，补齐在最后，补齐用的词汇默认是词汇表index=0的词汇，也可通过value指定
#     # 训练好的w2v词表的index = 0 对应的词汇是空格
#     X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH, padding='post')
#
#     np.save(outFile, np.column_stack([X, y]))
#
#     # 划分数据
#     Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # 构建网络
#     HIDDEN_LAYER_SIZE = 64
#     EMBEDDING_SIZE = 128
#
#     model = Sequential()
#     model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
#     model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(1))
#     model.add(Activation("sigmoid"))
#     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#
#     # 训练网络
#     BATCH_SIZE = 32
#     NUM_EPOCHS = 10
#     model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))
#
#     # 预测
#     score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
#     print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
#     print('{}   {}      {}'.format('预测', '真实', '句子'))
#     for i in range(5):
#         idx = np.random.randint(len(Xtest))
#         xtest = Xtest[idx].reshape(1, 70)
#         ylabel = ytest[idx]
#         ypred = model.predict(xtest)[0][0]
#         sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
#         print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
#
#
# def main():
#     #     inFile = "./data/test.txt"
#     inFile = "./data/processed_lstm/all_summary.txt"
#     outFile = "./model/train_lstm.npy"
#
#     # 把分词以后的文本转化为供LSTM训练的数据文件
#     trainLSTM(inFile, outFile)
#
# if __name__ == '__main__':
#     main()