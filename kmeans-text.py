from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import jieba
import matplotlib.pyplot as plt
import pathlib
import os

def wenjianjiajulei(datapath):
    content_series = []
    for fname_in in pathlib.Path(datapath).glob('*/'):
        f = open(fname_in, encoding="utf-8")
        str = f.read() #str放到一个
        if len(str)==0:
            continue
        content_series.append(str)

    corpus = []
    for i in range(len(content_series)):
        content = content_series[i]
        cutWords = [k for k in jieba.cut(content, True)]
        corpus.append(cutWords)

    table = ''.join(e for e in os.path.basename(datapath) if e.isalnum())
    cutWords = [k for k in jieba.cut(table, True)]
    basename_len = len(cutWords)
    corpus.append(cutWords)

    corpus = sum(corpus, [])
    print(corpus)

    vectorizer = CountVectorizer() # 将文本中的词语转换为词频矩阵，矩阵元素a[i][j]。表示j词在i类文本下的词频
    transformer = TfidfTransformer() # 统计每个词语的tf-idf权值
    print(vectorizer.fit_transform(corpus).toarray()) # 计算某个词出现的次数
    print(vectorizer.vocabulary_) # 查看词汇表
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus)) # #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()
    print("word feature length: {}".format(len(word)))
    tfidf_weight = tfidf.toarray()

    kmeans = KMeans(n_clusters=1)
    kmeans.fit(tfidf_weight)
    print(kmeans.cluster_centers_)
    # for index, label in enumerate(kmeans.labels_, 1):
    #     print("index: {}, label: {}".format(index, label))
    print("inertia: {}".format(kmeans.inertia_))

    tsne = TSNE(n_components=2) # 使用T-SNE算法，对权重进行降维
    decomposition_data = tsne.fit_transform(tfidf_weight)
    x = []
    y = []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    labels = kmeans.labels_
    labels[len(labels)-basename_len:] = 1
    # fig = plt.figure(figsize=(10, 10))
    # ax = plt.axes()
    plt.scatter(x, y, c=labels, marker="x") # 绘制散点图
    plt.xticks(())
    plt.yticks(())
    plt.show()
    # plt.savefig('./sample.png', aspect=1)

path = "C:\\Users\SQ\Desktop\实验文件\数据集\word\ocr"
for fname_in in pathlib.Path(path).glob('*/'):
    wenjianjiajulei(fname_in)