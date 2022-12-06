import numpy as np
import pandas as pd

from gensim import corpora
from gensim.models import ldamodel, TfidfModel, CoherenceModel
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s : ', level=logging.INFO)

data = pd.read_json('./datasets/data.json')
# 生成词典和语料
words = data['words'].to_list()
dictionary = corpora.Dictionary(words)
dictionary.filter_extremes(no_below=10) # 过滤低频词
dictionary.compactify()
corpus = [dictionary.doc2bow(text) for text in words]
tfidf = TfidfModel(corpus, dictionary=dictionary) 

# 超参搜索探索最佳主题数
def compute_coherence_values(dictionary, corpus, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = ldamodel.LdaModel(corpus=corpus, 
                            id2word=dictionary, num_topics=num_topics, passes=50)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, 
                        corpus=corpus, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
    # Print the coherence scores
    x = range(start, limit, step)
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    # Show graph
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig('./the_num_of_Topics.png')
    plt.show()
    return model_list, coherence_values

# model_list, coherence_values = compute_coherence_values(
#     dictionary=dictionary, corpus=tfidf[corpus], start=5, limit=500, step=5)

# LDA计算
n_topics = 95
lda = ldamodel.LdaModel(corpus=tfidf[corpus], 
                    id2word=dictionary, 
                    num_topics=n_topics, passes=50)
chapter_topics = lda.get_document_topics(corpus, minimum_probability=0.01)
mt_chapter_topics = np.zeros(shape=(120, n_topics), dtype=np.float32)
for i, topics in enumerate(chapter_topics):
    for j, prob in topics:
        mt_chapter_topics[i][j] = prob

# K-means聚类主题
kmeans = KMeans(n_clusters = 16, random_state=0).fit(mt_chapter_topics.T)
y_pred = [kmeans.cluster_centers_[i] for i in kmeans.predict(mt_chapter_topics.T)]
errors = [-mean_squared_error(mt_chapter_topics.T[i], y_pred[i]) 
            for i in range(n_topics)]
topic_words = [lda.show_topic(k, topn=20) for k in np.argsort(errors)[:3]]
topic_words = np.unique([word[0] for word in np.concatenate(topic_words)])
print(topic_words)
exit(0)
# 准备数据
words = np.concatenate(data.words).tolist()
vectorizer = CountVectorizer()
tf = vectorizer.fit_transform(words)

# LDA主题分析
n_topics = 200
lda = LatentDirichletAllocation(
    n_components=n_topics, 
    max_iter=50, 
    learning_method='online',                 
    learning_offset=50.)
lda.fit(tf)
# 得到每个章节属于某个主题的可能性
chapter_top = pd.DataFrame(
    lda.transform(tf),
    index=range(120),
    columns=np.arange(n_topics) + 1)
print(chapter_top)
exit(0)




# # 生成词典和语料
# words = data['words'].to_list()
# dictionary = corpora.Dictionary(words)
# corpus = [dictionary.doc2bow(text) for text in words]
# tf_idf = TfidfModel(corpus, dictionary=dictionary)[corpus]

# # LDA计算
# lda = ldamodel.LdaModel(corpus=tf_idf, id2word=dictionary, num_topics=200)


# 词频统计
words = np.concatenate(data.words)
words = pd.DataFrame({'words': words})
frequency = words.groupby(by=['words'])['words'].agg([('frequency', np.size)])
frequency = frequency.reset_index().sort_values(by='frequency', ascending=False)

# 构建语料库，创建TF-IDF矩阵
content = [' '.join(word) for word in data.words]
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(content)
word_vectors = tfidf.toarray()
# print(word_vectors)
# LDA分析主题
n_topics = 200
lda = LatentDirichletAllocation(
    n_components=n_topics, 
    max_iter=50, 
    learning_method='online',                 
    learning_offset=50.)
lda.fit(tfidf)
# 得到每个章节属于某个主题的可能性
chapter_top = pd.DataFrame(
    lda.transform(tfidf),
    index=range(120),
    columns=np.arange(n_topics) + 1)
print(chapter_top)