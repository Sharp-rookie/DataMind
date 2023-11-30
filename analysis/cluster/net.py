"""
聚类或者画网络关系图
也可以针对某个歌手分析
作者：谷泽昆
"""

import jieba
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import umap
import numpy as np
import string
import matplotlib.cm as cm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# 1. 准备数据
# 读取 Excel 数据
theme = '爱情'
singer = '郑冰冰'
mode = 'network'  # network or cluster
file_path = "data.csv" # f'主题/{theme}.csv'
df = pd.read_csv(file_path)
df = df[df['singer_name'] == singer]
emb_path = 'E:\LEARN\coding\Project\CNN\example_code\sgns.wiki.bigram'  # 修改为维基百科中文词向量预训练模型的路径
emb_size = 300
entity_model = KeyedVectors.load_word2vec_format(emb_path, binary=False, unicode_errors='ignore')


def get_word_vector(entity_model, word):
    if type(entity_model) == tuple:
        # 如果是个三元组，找出word对应在vocab中的id,再根据Id找出emb
        vocab, emb = entity_model
        wid = vocab[word]
        return emb[wid]
    else:
        if word in entity_model:
            return entity_model[word]
        else:
            emb = np.random.rand(emb_size).astype(np.float32)
            emb *= 0.1
            return emb


# 读取停用词文件
stop_words_file = 'ChineseStopWords.txt'
with open(stop_words_file, 'r', encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 添加自定义停用词
custom_stop_words = {'作词', '作曲', '贝斯', '收起', '制作', '编曲', '制作人', '和声', '混音', '钢琴', '鼓', '贝贝',
                     '箱琴', '笛子', '弦乐', '弦乐编写', '童声', 'Studio', '录音', '录音棚', '录音室', '录音师', '母带',
                     '监制', '网易', 'la', 'da', '', ' ', '…','副歌','主歌','冰冰','郑','间奏','结尾','演唱','后期',
                     '音乐','调','过渡'}
stop_words = stop_words.union(custom_stop_words).union(string.punctuation)

# 2. 分词
tokenized_lyrics = []
for lyric in df['lyrics']:
    words = jieba.lcut(lyric)
    filtered_words = [word for word in words if
                      word not in stop_words and not any(char in string.ascii_letters for char in word)]
    tokenized_lyrics.append(filtered_words)

# 3. 加载Word2Vec模型

# model = Word2Vec(sentences=tokenized_lyrics, vector_size=100, window=5, min_count=1, workers=4)
# model = Word2Vec.load('word2vec_model_with_stopwords_all_lyrics.model')
flatten_lyrics = sum(tokenized_lyrics, [])
flatten_lyrics_set = list(set(flatten_lyrics))

word_freq = Counter(flatten_lyrics)

# 输出词频最高的前 n 个词
n = 20
print(f"Top {n} words by frequency:")
for word, freq in word_freq.most_common(n):
    print(f"{word}: {freq}")

# region 聚类
if mode == 'cluster':
    # 获取每个词语的词向量
    # word_vectors = [model.wv[word] for word in model.wv.index_to_key]
    word_vectors = [get_word_vector(entity_model, word) for word in flatten_lyrics_set]
    # 转换为NumPy数组
    X = np.array(word_vectors)

    # 使用K均值聚类
    num_clusters = [2,3,4,5,6,7,8]  # 设置聚类的数量
    for num_cluster in num_clusters:
        kmeans = KMeans(n_clusters=num_cluster,random_state=42)
        kmeans.fit(X)

        # 获取每个类别的中心
        cluster_centers = kmeans.cluster_centers_

        # 获取每个词语所属的聚类
        labels = kmeans.labels_

        top_words_per_cluster = []
        flatten_lyrics_set_np = np.array(flatten_lyrics_set)
        for i in range(num_cluster):
            # 计算每个词语与中心的余弦相似度
            similarities = cosine_similarity(X[labels == i], [cluster_centers[i]])
            # 获取相似度最高的前5个词语的索引
            top_word_indices = similarities.flatten().argsort()[-5:][::-1]
            # 获取对应的词语

            top_words = [flatten_lyrics_set_np[labels == i][index] for index in top_word_indices]
            top_words_per_cluster.append(top_words)


        # 输出每个类别的前5个词语
        for i, top_words in enumerate(top_words_per_cluster):
            print(f"Cluster {i}: {', '.join(top_words)}")

        # 可视化聚类结果（使用PCA降维）
        myumap = umap.UMAP(n_components=2)
        X_tsne = myumap.fit_transform(X)

        plt.figure(figsize=(10, 6))
        for i in range(num_cluster):
            cluster_points = X_tsne[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

            # 显示每个类别的前5个词语标签
            for word in top_words_per_cluster[i]:
                index = flatten_lyrics_set.index(word)
                plt.annotate(word, (X_tsne[index, 0], X_tsne[index, 1]), textcoords="offset points")

        plt.title(f'Word Clusters {num_cluster}')
        plt.legend()
        plt.show()
# endregion

# region 网络图
if mode == 'network':
    # 4. 计算高频词之间的欧氏距离
    top_words = [word for word, _ in word_freq.most_common(n)]
    # word_vectors = {word: model.wv[word] for word in top_words}
    word_vectors = {word: get_word_vector(entity_model, word) for word in top_words}

    # 创建无向图
    G = nx.Graph()

    # 添加节点和边，只添加距离较近的边
    # TODO: 需要根据具体情况调整一下阈值
    thres = {
        '爱情': 0.1,
        '成长': 0.095,
        '励志': 0.1,
        '教育': 0.1,
        '红色': 0.2
    }
    node_S = {
        '爱情': 5,
        '红色': 20,
        '教育': 20,
        '成长': 20,
        '励志': 20
    }
    line_S = {
        '爱情': 100.,
        '红色': 50.,
        '教育': 50.,
        '成长': 50.,
        '励志': 50.
    }
    threshold = thres[theme]  # 设置距离阈值，根据具体数据调整
    for i in range(n):
        for j in range(i + 1, n):
            word1, word2 = top_words[i], top_words[j]
            similarity = entity_model.similarity(word1, word2)  # 余弦相似度
            # distance = np.linalg.norm(word_vectors[word1] - word_vectors[word2])  # 欧氏距离
            if similarity > threshold:
                G.add_node(word1)
                G.add_node(word2)
                G.add_edge(word1, word2, weight=similarity * 10)  # 注意这里权重的计算，可以根据具体需求调整

    # 5. 绘制网络图
    pos = nx.spring_layout(G, seed=15)
    labels = {word: word for word in G.nodes()}
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # 计算节点的大小，可以根据词频调整大小比例
    node_size = node_S[theme]
    node_sizes = [word_freq[word] * node_size for word in G.nodes()]

    # 计算边的粗细，可以根据词频调整粗细比例
    line_scale=line_S[theme]
    edge_widths = [(word_freq[u] + word_freq[v]) / line_scale for u, v in G.edges()]

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=cm.Wistia, width=edge_widths, alpha=0.7)

    # 添加权重颜色条
    cbar = plt.colorbar(edges)
    cbar.set_label('Edge Weight', rotation=270, labelpad=15)

    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black', font_family='SimHei')
    plt.title(f'High-Frequency Words Network with Euclidean Distance (Threshold < {threshold})')
    plt.show()
# endregion
