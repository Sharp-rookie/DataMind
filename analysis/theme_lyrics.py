import os
import jieba.posseg as psg
import pdb
from PIL import Image, ImageSequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import jieba
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fmgr
import brewer2mpl


STOP_PATH = './ChineseStopWords.txt'
THEME_DIR = './主题/'

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

class Theme(object):
    def __init__(self, theme_path, stop_words) -> None:
        self.df = pd.read_csv(theme_path)
        self.stop_words = stop_words
    
    def get_title(self):
        title_list = self.df['song'].values.tolist()
        return title_list
    
    def get_lyrics(self):
        lyrics_list = self.df['lirics'].values.tolist()
        return lyrics_list

    def word_count(self, text_list):
        words_dict = {}
        jieba.add_word('梦想')
        for text in text_list:
            try:
                new_text = allcontents(text)
            except Exception:
                continue
            for word in jieba.cut(new_text, cut_all=False):
                if word in self.stop_words or word == ' ':
                    continue
                if word in words_dict.keys():
                    words_dict[word] += 1
                else:
                    words_dict[word] = 1
        
       
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
        top_n = 10
        word_count_sorted = sorted(filter_dict.items(), key=lambda x: x[1], reverse=True)
        word_top_n = word_count_sorted[:top_n]
        words = [item[0] for item in word_top_n]
        counts = [item[1] for item in word_top_n]

        YaHei = fmgr.FontProperties(fname='./SimHei.ttf')
        plt.figure(figsize=(10, 10))
        plt.bar(words, counts)
        plt.xticks(rotation=45, fontproperties=YaHei, fontsize=25)
        plt.yticks(fontsize=20)
        plt.savefig(name+'_bar.jpg')
        
    def build_cloud(self, fig_name, is_title=False):
        if is_title:
            words_dict = self.word_count(self.get_title())
        else:
            words_dict = self.word_count(self.get_lyrics())
        image = Image.open("./img5.png")
        graph = np.array(image)
        wc = WordCloud(mask=graph, background_color = 'white', font_path='./SimHei.ttf', max_words=200, max_font_size=150)
        wc.generate_from_frequencies(words_dict)
        image_color = ImageColorGenerator(graph)
        plt.imshow(wc.recolor(color_func=image_color)) #对词云重新着色
        plt.axis('off')
        wc.to_file(fig_name + ".png")
        plt.clf()


def main():
    stop_words = set()
    with open('./ChineseStopWords.txt', 'r') as f:
        for line in f.readlines():
            stop_words.add(line.strip())
    stop_words = list(stop_words)
    files = os.listdir(THEME_DIR)
    for file in files:
        theme = Theme(
            theme_path=os.path.join(THEME_DIR, file),
            stop_words=stop_words
        )
        theme.build_bar(file.strip().split('.')[0])


main()