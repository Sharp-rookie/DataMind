# 分年份或者歌手计算共现矩阵
# 作者：谷泽昆

import pandas as pd
from collections import Counter
import jieba
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# 读取数据
df = pd.read_csv('data.csv')

# 假设我们有一个特定歌手的出生日期
birthdate = '1980'
# singer_name = '周杰伦'

# 计算开始的年代
start_year = (int(birthdate[:4]) // 10) * 10

# 过滤出特定歌手的数据
# df_singer = df[df['singer_name'] == singer_name]

# 读取停用词文件
stop_words_file = 'ChineseStopWords.txt'
with open(stop_words_file, 'r', encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 添加自定义停用词
custom_stop_words = {'作词', '作曲', '贝斯', '收起', '制作', '编曲', '制作人', '和声', '混音', '钢琴', '鼓', '贝贝',
                     '箱琴', '笛子', '弦乐', '弦乐编写', '童声', 'Studio', '录音', '录音棚', '录音室', '录音师', '母带',
                     '监制', '网易', 'la', 'da', '', ' ', '…', '副歌', '主歌', '冰冰', '郑', '间奏', '结尾', '演唱',
                     '后期', '音乐', '调', '过渡', '\xa0', '\u3000', '\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
                     '说', '想', '爱', '一生', '爱情', '请', '问', '不', '没', '心中', '做', '更', '为什', '什', '倒',
                     '傥', '元', '先', '兼', '前', '吨', '唷', '啪', '啷', '喔', '声', '外', '多年', '大面儿', '天',
                     '始', '常', '後', '抗拒', '敞开', '数', '新', '方', '日', '昉', '末', '次', '毫无保留', '没', '漫',
                     '然', '特', '特别', '理', '皆', '目前为止', '竟', '策略', '编写', '莫', '见', '设', '话', '赶早',
                     '赶晚', '达', '限', '非', '面', '麽','走','中','太','里'}
stop_words = stop_words.union(custom_stop_words).union(string.punctuation)


# 分词和计算共现矩阵的函数
def calculate_co_occurrence(data):
    # 分词
    tokenized_lyrics = []
    for lyric in data['lyrics']:
        try:
            words = jieba.lcut(lyric)
            filtered_words = [word for word in words if
                              word not in stop_words and not any(char in string.ascii_letters for char in word)]
            tokenized_lyrics.append(filtered_words)
        except AttributeError as e:
            continue

    flatten_lyrics = sum(tokenized_lyrics, [])
    word_freq = Counter(flatten_lyrics)

    # 输出词频最高的前 n 个词
    n = 15
    top_words = [word for word, freq in word_freq.most_common(n)]

    # Initialize a co-occurrence matrix
    co_occurrence_matrix = np.zeros((len(top_words), len(top_words)), dtype=int)

    # Building the co-occurrence matrix
    for i, word1 in enumerate(top_words):
        for j, word2 in enumerate(top_words):
            if i != j:
                # Count the number of times word1 and word2 appear in the same lyric
                co_occurrence_matrix[i, j] = sum(
                    lyric.count(word1) > 0 and lyric.count(word2) > 0 for lyric in tokenized_lyrics)

    # Plotting the co-occurrence matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=top_words, yticklabels=top_words)
    plt.title(f"Co-occurrence Matrix of Top 15 Words for ({decade_start}-{decade_end}s)")
    plt.xlabel("Words")
    plt.ylabel("Words")
    plt.show()


# 循环处理每个十年期间
interval = 2
for decade_start in range(start_year, 2040, interval):
    print(f"Processing decade: {decade_start}s")
    decade_end = decade_start + interval - 1
    df_decade = df[(df['Year'] >= decade_start) & (df['Year'] <= decade_end)]
    if df_decade.shape[0] == 0:
        continue
    calculate_co_occurrence(df_decade)
