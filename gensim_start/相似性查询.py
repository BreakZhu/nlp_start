# -*— encoding: utf-8 -*-
import logging
import os
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
之前我们介绍了在向量空间模型中创建语料库以及如何在不同向量空间之间进行转换的含义。这种特征的一个常见原因是我们想要确定 
文档对之间的相似性，或者特定文档与一组其他文档（例如用户查询与索引文档）之间的相似性。
为了说明在gensim中如何做到这一点，让我们考虑与之前的例子相同的语料库
（它最初来自Deerwester等人的“潜在语义分析索引” 1990年开篇 文章）：
"""
dictionary = corpora.Dictionary.load('../model/deerwester.dict')
corpus = corpora.MmCorpus('../model/deerwester.mm')  # comes from the first tutorial, "From strings to vectors"
print corpus

# 用这个小的语料库来定义一个二维LSI空间
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
# 现在假设用户输入查询“Human computer interaction”。我们希望按照与此查询相关的递减顺序对我们的九个语料库文档进行排序。
# 与现代搜索引擎不同 这里我们只关注可能相似性的一个方面
# 关于其文本（单词）的明显语义相关性。没有超链接， 没有随机游走静态排名，只是布尔关键字匹配的语义扩展：
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
print vec_lsi
# 我们将考虑余弦相似性 来确定两个向量的相似性。余弦相似度是向量空间建模中的标准度量，但是无论向量表示概率分布，
# 不同的相似性度量 可能更合适

# 初始化查询结构
# 为了准备相似性查询，我们需要输入我们想要与后续查询进行比较的所有文档。在我们的例子中，
# 它们与用于训练LSI的九个文件相同，转换为二维LSA空间。但这只是偶然的，我们也可能完全索引不同的语料库
index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
"""
注意：
similarities.MatrixSimilarity只有当整个向量集适合内存时，该类才适用。例如，当与此类一起使用时，
一百万个文档的语料库在256维LSI空间中将需要2GB的RAM。如果没有2GB的可用RAM，则需要使用similarities.Similarity该类。
此类通过在磁盘上的多个文件（称为分片）之间拆分索引，在固定内存中运行。
它使用similarities.MatrixSimilarity和similarities.SparseMatrixSimilarity内部，所以它仍然很快，虽然稍微复杂一点。
索引持久性通过标准save()和load()函数处理：
"""
index.save('../model/deerwester.index')
index = similarities.MatrixSimilarity.load('../model/deerwester.index')
"""
对于所有相似性索引类（similarities.Similarity， similarities.MatrixSimilarity和similarities.SparseMatrixSimilarity）
都是如此。同样在下文中，索引可以是任何这些的对象。如果有疑问，请使用similarities.Similarity，因为它是最具扩展性的版本，
并且它还支持稍后向索引添加更多文档
"""
# 执行查询
sims = index[vec_lsi]  # perform a similarity query against the corpus
print list(enumerate(sims))  # print (document_number, document_similarity) 2-tuples
# 余弦测量返回范围<-1,1>中的相似度（越大，越相似）
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print sims  # print sorted (document number, similarity score) 2-tuples
"""
这里要注意的是文件没有。标准布尔全文搜索永远不会返回2（）和4（），因为它们不共享任何常用单词。然而，在应用LSI之后，
我们可以观察到它们都获得了相当高的相似性得分（第2类实际上是最相似的！），这更符合我们对它们与查询共享“计算机 - 人”
相关主题的直觉。事实上，这种语义概括是我们首先应用转换并进行主题建模的原因
"""
