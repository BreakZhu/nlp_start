# -*— encoding: utf-8 -*-
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

# 在gensim中，语料库只是一个对象，当迭代时，返回其表示为稀疏向量的文档。在这种情况下，我们使用元组列表的列表。
# 如果您不熟悉矢量空间模型，我们将在下一个关于Corpora和Vector Spaces的教程中弥合原始字符串，语料库和稀疏矢量之间的差距。
# 如果您熟悉向量空间模型，您可能会知道解析文档并将其转换为向量的方式会对任何后续应用程序的质量产生重大影响。
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)], [(9, 1.0)],
          [(9, 1.0), (10, 1.0)], [(9, 1.0), (10, 1.0), (11, 1.0)], [(8, 1.0), (10, 1.0), (11, 1.0)]]
# 在此示例中，整个语料库作为Python列表存储在内存中。但是，语料库接口只表示语料库必须支持对其组成文档的迭代。
# 对于非常大的语料库，有利的是将语料库保持在磁盘上，并且一次一个地顺序访问其文档。
# 所有操作和转换都以这样的方式实现，使得它们在内存方面独立于语料库的大小。
# 接下来，让我们初始化一个转换
tfidf = models.TfidfModel(corpus)
vec = [(0, 1), (4, 1)]
print tfidf[vec]
# 在这里，我们使用了Tf-Idf，这是一种简单的转换，它将文档表示为词袋计数，并应用对常用术语进行折扣的权重
# （或者等同于促销稀有术语）。它还将得到的向量缩放到单位长度（在欧几里德范数中）。主题和转换教程中详细介绍了转换。
# 要通过TfIdf转换整个语料库并对其进行索引，以准备相似性查询：
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
# 并查询我们的查询向量vec与语料库中每个文档的相似性：
sims = index[tfidf[vec]]
print list(enumerate(sims))
"""
如何阅读此输出？文档编号为零（第一个文档）的相似度得分为0.466 = 46.6％，第二个文档的相似度得分为19.1％等。
因此，根据TfIdf文档表示和余弦相似性度量，最类似于我们的查询文档vec是文档号。3，相似度得分为82.1％。请注意，
在TfIdf表示中，任何不具有任何共同特征的vec文档（文档编号4-8）的相似性得分均为0.0
"""