import jieba
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm

file_paths = ['data.csv' ,'主题/爱情.csv', '主题/红色.csv', '主题/教育.csv', '主题/成长.csv', '主题/励志.csv']
tokenized_lyrics = []

# 读取停用词文件
stop_words_file = 'ChineseStopWords.txt'
with open(stop_words_file, 'r', encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f])

# 添加自定义停用词
custom_stop_words = {'作词', '作曲', '贝斯', '收起', '制作', '编曲', '制作人', '和声', '混音', '钢琴', '鼓', '贝贝',
                     '箱琴', '笛子', '弦乐', '弦乐编写', '童声', 'Studio', '录音', '录音棚', '录音室', '录音师',
                     '母带', '监制', '网易', 'la', 'da', '', ' ', '…'}
stop_words = stop_words.union(custom_stop_words)

for file_path in tqdm(file_paths, desc="Processing Files"):
    df = pd.read_csv(file_path)

    # 2. 分词
    for lyric in tqdm(df['lyrics'], desc="Processing Lyrics", leave=False):
        try:
            words = jieba.lcut(lyric)
        except AttributeError as e:
            continue
        filtered_words = [word for word in words if word not in stop_words]
        tokenized_lyrics.append(filtered_words)

print('begin train')
# 3. 训练Word2Vec模型
model = Word2Vec(sentences=tokenized_lyrics, vector_size=100, window=5, min_count=1, workers=4)

# 4. 保存模型
model.save('word2vec_model_with_stopwords_all_lyrics.model')

