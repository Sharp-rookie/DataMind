"""
计算某个主题的歌词的共现矩阵
作者：谷泽昆
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from collections import Counter
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

theme = '教育'
file_path = f'主题/{theme}.csv'
df = pd.read_csv(file_path)

# 读取停用词文件
stop_words_file = 'ChineseStopWords.txt'
with open(stop_words_file, 'r', encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 添加自定义停用词
custom_stop_words = {'作词', '作曲', '贝斯', '收起', '制作', '编曲', '制作人', '和声', '混音', '钢琴', '鼓', '贝贝',
                     '箱琴', '笛子', '弦乐', '弦乐编写', '童声', 'Studio', '录音', '录音棚', '录音室', '录音师', '母带',
                     '监制', '网易', 'la', 'da', '', ' ', '…', '副歌', '主歌', '冰冰', '郑', '间奏', '结尾', '演唱',
                     '后期', '音乐', '调', '过渡'}
stop_words = stop_words.union(custom_stop_words).union(string.punctuation)

# 2. 分词
tokenized_lyrics = []
for lyric in df['lyrics']:
    words = jieba.lcut(lyric)
    filtered_words = [word for word in words if
                      word not in stop_words and not any(char in string.ascii_letters for char in word)]
    tokenized_lyrics.append(filtered_words)

flatten_lyrics = sum(tokenized_lyrics, [])
flatten_lyrics_set = list(set(flatten_lyrics))

word_freq = Counter(flatten_lyrics)

# 输出词频最高的前 n 个词
n = 15
top_words = []
print(f"Top {n} words by frequency:")
for word, freq in word_freq.most_common(n):
    top_words.append(word)
    print(f"{word}: {freq}")

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
plt.title("Co-occurrence Matrix of Top 10 Words")
plt.xlabel("Words")
plt.ylabel("Words")
plt.show()
