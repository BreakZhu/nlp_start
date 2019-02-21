# -*— encoding: utf-8 -*-
import logging
import os
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 转换接口
# 在上一篇关于Corpora和Vector Spaces的教程中，我们创建了一个文档语料库，表示为向量流。
# 要继续，让我们启动gensim并使用该语料库：
if os.path.exists("../model/deerwester.dict"):
    dictionary = corpora.Dictionary.load('../model/deerwester.dict')
    corpus = corpora.MmCorpus('../model/deerwester.mm')
    print "Used files generated from first tutorial"
else:
    print "Please run first tutorial to generate data set"

"""
我将展示如何将文档从一个矢量表示转换为另一个矢量表示。这个过程有两个目标：
为了在语料库中显示隐藏的结构，发现单词之间的关系并使用它们以新的（希望）更加语义的方式描述文档。
使文档表示更紧凑。这既提高了效率（新表示消耗更少的资源）和功效（忽略边际数据趋势，降低噪音）。
"""
# 创建转换
# 转换是标准的Python对象，通常通过训练语料库进行初始化
# step 1 -- initialize a model
tfidf = models.TfidfModel(corpus)
print tfidf
"""
我们使用教程1中的旧语料库来初始化（训练）转换模型。不同的转换可能需要不同的初始化参数; 
在TfIdf的情况下，“训练”仅包括通过提供的语料库一次并计算其所有特征的文档频率。
训练其他模型，例如潜在语义分析或潜在Dirichlet分配，涉及更多，因此需要更多时间。
转换总是在两个特定的向量空间之间转换。必须使用相同的向量空间（=同一组特征id）进行训练以及后续的向量转换。
无法使用相同的输入要素空间，例如应用不同的字符串预处理，使用不同的特征ID，或使用预期为TfIdf向量的词袋输入向量，
将导致转换调用期间的特征不匹配，从而导致垃圾中的任何一个输出和/或运行时异常
"""
# 变换向量
# 从现在开始，tfidf被视为一个只读对象，可用于将任何向量从旧表示（bag-of-words整数计数）转换为新表示（TfIdf实值权重）
doc_bow = [(0, 1), (1, 1)]
print tfidf[doc_bow]  # step 2 -- use the model to transform vectors
# 或者将转换应用于整个语料库
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print doc

# 在这种特殊情况下，我们正在改变我们用于训练的同一语料库，但这只是偶然的。一旦初始化了转换模型， 它就可以用于任何向量
# （当然，只要它们来自相同的向量空间），即使它们根本没有在训练语料库中使用。
# 这是通过LSA的折叠过程，LDA的主题推断等来实现的 调用model[corpus]仅在旧corpus 文档流周围创建一个包装器-
# 实际转换在文档迭代期间即时完成。我们无法在调用时转换整个语料库，因为这意味着将结果存储在主存中，这与gensim
# 的内存独立目标相矛盾。如果您将多次迭代转换，并且转换成本很高，请先将生成的语料库序列化为磁盘并继续使用它。
# corpus_transformed = model[corpus]corpus_transformed

# 转换也可以序列化，一个在另一个之上，在一个链中
# 在这里，我们通过潜在语义索引将我们的Tf-Idf语料库 转换为潜在的2-D空间（因为我们设置了2-D num_topics=2）。
# 现在你可能想知道：这两个潜在的维度代表什么？让我们检查一下models.LsiModel.print_topics()：
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(2)
"""
根据LSI，“树”，“图”和“未成年人”似乎都是相关词（并且对第一个主题的方向贡献最大），
而第二个主题实际上与所有其他词有关。正如所料，前五个文件与第二个主题的关联性更强，而剩下的四个文件与第一个主题相关：
"""
for doc in corpus_lsi:  # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)
# 使用save()和load()函数实现模型持久性：
lsi.save('../model/model.lsi')  # same for tfidf, lda, ...
lsi = models.LsiModel.load('../model/model.lsi')

# 可用的转换
# Gensim实现了几种流行的矢量空间模型算法：

# Term Frequency * Inverse Document Frequency   TF-IDF
"""
期望初始化期间的词袋（整数值）训练语料库。在变换期间，它将采用向量并返回具有相同维度的另一个向量，
除了在训练语料库中罕见的特征将增加其值。因此，它将整数值向量转换为实值向量，同时保持维度的数量不变。
它还可以任选地将得到的矢量归一化为（欧几里得）单位长度
"""
model = models.TfidfModel(corpus, normalize=True)

# Latent Semantic Indexing, LSI (or sometimes LSA)
"""
将文档从单词袋或（优选地）TfIdf加权空间转换为较低维度的潜在空间。对于上面的玩具语料库，我们只使用了2个潜在维度，
但在实际语料库中，建议将200-500的目标维度作为“黄金标准”
LSI培训的独特之处在于我们可以随时继续“培训”，只需提供更多培训文件即可。
这是通过在称为在线培训的过程中对底层模型的增量更新来完成的。由于这个特性，输入文档流甚至可能是无限的 
- 只需在LSI新文档到达时继续提供它们，同时使用计算的转换模型作为只读
"""
model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
# model.add_documents(another_tfidf_corpus) # now LSI has been trained on tfidf_corpus + another_tfidf_corpus
# lsi_vec = model[tfidf_vec] # convert some new document into the LSI space, without affecting the model
# model.add_documents(more_documents) # tfidf_corpus + another_tfidf_corpus + more_documents
# lsi_vec = model[tfidf_vec]

# Random Projections, RP
"""
旨在减少向量空间维度。这是一种非常有效的（内存和CPU友好的）方法，通过投入一点随机性来近似文档之间的TfIdf距离。
建议的目标维度再次为数百/数千，具体取决于您的数据集。
"""
model = models.RpModel(corpus_tfidf, num_topics=500)

# Latent Dirichlet Allocation . LDA
"""
 是另一种从词袋计数转变为低维度主题空间的转变。LDA是LSA（也称为多项PCA）的概率扩展，
 因此LDA的主题可以解释为对单词的概率分布。与LSA一样，这些分布也是从训练语料库中自动推断出来的。
 文档又被解释为这些主题的（软）混合（再次，就像LSA一样）
"""
model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)

# Hierarchical Dirichlet Process, HDP   是一种非参数贝叶斯方法（请注意缺少的请求主题数）
"""
HDP模型是gensim的新成员，并且在学术方面仍然很粗糙 - 谨慎使用
"""
model = models.HdpModel(corpus, id2word=dictionary)
