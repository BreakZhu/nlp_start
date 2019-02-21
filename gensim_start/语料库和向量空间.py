# -*— encoding: utf-8 -*-
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 从字符串到向量 这一次，让我们从表示为字符串的文档开始：
from gensim import corpora

# 从字符串到向量
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
# 这是一个由九个文档组成的小型语料库，每个文档只包含一个句子。
# 首先，让我们对文档进行标记，删除常用单词 以及仅在语料库中出现一次的单词：
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
# remove words that appear only once
from collections import defaultdict

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint  # pretty-printer

pprint(texts)
"""
处理文档的方式是多种多样的，依赖于应用程序和语言，我决定不通过任何接口约束它们。相反，文档由从中提取的特征表示，
而不是由其“表面”字符串形式表示：如何使用这些特征取决于您。下面我描述一种常见的通用方法（称为 词袋），但请记住，
不同的应用程序域需要不同的功能要将文档转换为向量，我们将使用名为bag-of-words的文档表示 。
在此表示中，每个文档由一个向量表示，其中每个向量元素表示问题 - 答案对，格式为：
“单词系统出现在文档中的次数是多少？一旦。”
仅通过它们的（整数）id来表示问题是有利的。问题和ID之间的映射称为字典：

"""
dictionary = corpora.Dictionary(texts)
dictionary.save('../model/deerwester.dict')  # store the dictionary, for future reference
print dictionary
"""
在这里，我们为语料库中出现的所有单词分配了一个唯一的整数id gensim.corpora.dictionary.Dictionary。
这会扫描文本，收集字数和相关统计数据。最后，我们看到处理后的语料库中有12个不同的单词，
这意味着每个文档将由12个数字表示（即，通过12-D向量）。要查看单词及其ID之间的映射：
{u'minors': 11, u'graph': 10, u'system': 5, u'trees': 9, u'eps': 8, u'computer': 0, u'survey': 4, u'user': 7,
 u'human': 1, u'time': 6, u'interface': 2, u'response': 3}
"""
print dictionary.token2id
# 要将标记化文档实际转换为向量：
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
# [(0, 1), (1, 1)]
"""
该函数doc2bow()只计算每个不同单词的出现次数，将单词转换为整数单词id，并将结果作为稀疏向量返回。 因此，稀疏向量读取：
在文档“Human computer interaction”中，单词computer （id 0）和human（id 1）出现一次; 其他十个字典单词（隐含地）出现零次。[(0, 1), (1, 1)]
"""
print new_vec
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('../model/deerwester.mm', corpus)  # store to disk, for later use
print corpus


#  语料库流 - 一次一个文档
class MyCorpus(object):
    def __iter__(self):
        """
        假设每个文档占用单个文件中的一行并不重要; 您可以模拟__iter__函数以适合您的输入格式，无论它是什么。
        行走目录，解析XML，访问网络 只需解析输入以在每个文档中检索一个干净的标记列表，
        然后通过字典将标记转换为它们的ID，并在__iter__中生成生成的稀疏向量
        :return:
        """
        for line in open('../dic/mycorpus'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())


corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print corpus_memory_friendly
# 语料库现在是一个对象。我们没有定义任何打印方式，因此print只输出内存中对象的地址。不是很有用。
# 要查看构成向量，让我们遍历语料库并打印每个文档向量（一次一个）：
for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print vector
"""
尽管输出与普通Python列表的输出相同，但语料库现在更加内存友好，因为一次最多只有一个向量驻留在RAM中。
您的语料库现在可以随意扩展。类似地，构造字典而不将所有文本加载到内存中：
"""
# 重构如下
from six import iteritems

# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('../dic/mycorpus'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print dictionary.token2id

# 语料库格式
# 存在几种用于将Vector Space语料库（〜矢量序列）序列化到磁盘的文件格式。
# Gensim通过前面提到的流式语料库接口实现它们：文件以懒惰的方式从（分别存储到）磁盘读取，
# 一次一个文档，而不是一次将整个语料库读入主存储器

# create a toy corpus of 2 documents, as a plain Python list
corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it
corpora.MmCorpus.serialize('../model/corpus.mm', corpus)
# 其他格式包括Joachim的SVMlight格式， Blei的LDA-C格式和 GibbsLDA ++格式。
corpora.SvmLightCorpus.serialize('../model/corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('../model/corpus.lda-c', corpus)
corpora.LowCorpus.serialize('../model/corpus.low', corpus)
# 相反，要从Matrix Market文件加载语料库迭代器：
corpus = corpora.MmCorpus('../model/corpus.mm')
# 要查看语料库的内容：
#  one way of printing a corpus: load it entirely into memory
print(list(corpus))  # calling list() will convert any sequence to a plain Python list
# another way of doing it: print one document at a time, making use of the streaming interface
for doc in corpus:
    print doc

# 与NumPy和SciPy的兼容性
import gensim
import numpy as np

# Gensim还包含有效的实用程序函数 来帮助转换为/ numpy矩阵：
numpy_matrix = np.random.randint(10, size=[5, 2])  # random matrix as an example
print numpy_matrix
corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
print list(corpus)
numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=5)
print numpy_matrix

# 从/到scipy.sparse矩阵
import scipy.sparse
scipy_sparse_matrix = scipy.sparse.random(5, 2)  # random sparse matrix as example
corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)
print scipy_csc_matrix
