import os
import jieba.posseg as psg
from PIL import Image, ImageSequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import jieba
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmgr

ERA_DIR = './年代/'
STOP_PATH = './ChineseStopWords.txt'

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # 判断一个uchar是否是汉字
        return True
    else:
        return False

def allcontents(contents):
    content = ''
    for i in contents:
        if is_chinese(i):
            content = content+i
    return content

def time_split():
    df = pd.read_csv('./time_lyrics.csv')
    time_list = [1980, 1990, 2000, 2010, 2020]
    for t in time_list:
        if t == 2020:
            new_df = df[(df['Year'] >= t)]
        else:
            new_df = df[(df['Year'] >= t)&(df['Year'] < (t+10))]
        new_df.to_csv(os.path.join(ERA_DIR, f"{t}.csv"))

class Era(object):
    def __init__(self, era_path, stop_words) -> None:
        self.df = pd.read_csv(era_path)
        self.stop_words = stop_words
        
    def get_title(self):
        title_list = self.df['song_name'].values.tolist()
        return title_list
    
    def get_lyrics(self):
        lyrics_list = self.df['lyric'].values.tolist()
        return lyrics_list

    def word_count(self, text_list):
        words_dict = {}
        for text in text_list:
            try:
                new_text = allcontents(text)
            except Exception:
                continue
            try:
                for word in jieba.cut(new_text, cut_all=False):
                    if word in self.stop_words or word == ' ':
                        continue
                    if word in words_dict.keys():
                        words_dict[word] += 1
                    else:
                        words_dict[word] = 1
            except Exception as e:
                continue

        return words_dict

    def build_bar(self, name, is_title=False):
        if is_title:
            words_dict = self.word_count(self.get_title())
        else:
            words_dict = self.word_count(self.get_lyrics())
        filter_dict = {}
        for key in list(words_dict.keys()):
            if len(key) == 1:
                continue
            filter_dict[key] = words_dict[key]

        word_count_sorted = sorted(filter_dict.items(), key=lambda x: x[1], reverse=True)
        word_top_n = word_count_sorted[30:50]
        words = [item[0] for item in word_top_n]
        counts = [item[1] for item in word_top_n]
        YaHei = fmgr.FontProperties(fname='./SimHei.ttf')
        plt.figure(figsize=(15, 10))
        plt.bar(words, counts)
        plt.xlabel('Words')
        plt.ylabel('Counts')
        plt.xticks(rotation=45, fontproperties=YaHei, fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(name+'_bar.jpg')
        
    def build_cloud(self, fig_name, is_title=False):
        if is_title:
            words_dict = self.word_count(self.get_title())
        else:
            words_dict = self.word_count(self.get_lyrics())

        filter_dict = {}
        for key in list(words_dict.keys()):
            if len(key) == 1:
                continue
            filter_dict[key] = words_dict[key]
        image = Image.open("./img4.png")
        graph = np.array(image)
        wc = WordCloud(mask=graph, background_color = 'white', font_path='./SimHei.ttf', max_words=200, max_font_size=150)
        wc.generate_from_frequencies(filter_dict)
        image_color = ImageColorGenerator(graph)
        plt.imshow(wc.recolor(color_func=image_color)) #对词云重新着色
        plt.axis('off')
        wc.to_file(fig_name + ".png")
        plt.clf()


def main():
    # time_split()
    stop_words = set()
    with open('./ChineseStopWords.txt', 'r') as f:
        for line in f.readlines():
            stop_words.add(line.strip())
    stop_words = list(stop_words)
    files = os.listdir(ERA_DIR)
    for file in files:
        print(file)
        era = Era(
            era_path=os.path.join(ERA_DIR, file),
            stop_words=stop_words
        )
        era.build_bar(file.strip().split('.')[0])


main()

