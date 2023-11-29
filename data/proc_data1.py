import json
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


class Song:
    def __init__(self, name, lyric):
        self.name = name
        self.lyric = lyric

class Singer:
    def __init__(self, name):
        self.name = name
        self.songs = []
    
    def add_song(self, song):
        self.songs.append(song)

class Singers:
    def __init__(self):
        self.singers_num = 0
        self.singers_name = []
        self.singers = []
    
    def add_singer(self, singer):
        self.singers.append(singer)
        self.singers_num += 1
        self.singers_name.append(singer.name)
    
    def add_song(self, singer_name, song):
        for i in range(self.singers_num):
            if self.singers[i].name == singer_name:
                self.singers[i].songs.append(song)
                break
        
    def filter_by_songs_num(self, num):
        singers = []
        singers_name = []
        for singer in self.singers:
            if len(singer.songs) >= num:
                singers.append(singer)
                singers_name.append(singer.name)
        self.singers = singers
        self.singers_num = len(singers)
        self.singers_name = singers_name
        
        print(f'\n{self.singers_num} singers have more than {num} songs.')
    
    def save(self):
        data = []
        singer_name = []
        for singer in self.singers:
            singer_name.append([singer.name])
            for song in singer.songs:
                data.append([singer.name, song.name, song.lyric])
        df = pd.DataFrame(data, columns=['singer_name', 'song_name', 'lyric'])
        df.to_csv(f'filtered.csv', index=False, encoding='utf_8_sig')
        df2 = pd.DataFrame(singer_name, columns=['singer_name'])
        df2.to_csv(f'singer_name.csv', index=False, encoding='utf_8_sig')
        print(f'\nSave data successfully.')
        
        
def get_birthdates(singer_list):
    "自动从百度百科中提取歌手的出生日期，并保存到 birthdates.csv 文件中。"
    
    birthdates = []
    
    # 创建Chrome WebDriver
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--log-level=3')
    # chrome_options.add_argument('--headless') # 无头模式，不显示浏览器窗口
    chrome_options.add_argument('--blink-settings=imagesEnabled=false') # 不加载图片，加快访问速度
    chrome_options.add_argument('--autoplay-policy=no-user-gesture-required') # 禁止自动播放
    driver = webdriver.Chrome(options=chrome_options)
    
    iter = tqdm(singer_list)
    for singer_name in iter:
        
        # 访问百度百科首页
        driver.get('https://baike.baidu.com/')
        
        # 在搜索框中输入歌手名并点击搜索按钮
        search_input = driver.find_element(By.ID, 'query')
        search_input.clear()
        search_input.send_keys(singer_name)
        search_input.send_keys(Keys.RETURN)
        
        # 获取搜索结果页面的内容
        elems = driver.find_elements(By.NAME, 'description')
        content = elems[0].get_attribute('content')
        
        # 使用正则表达式匹配出生日期
        result = re.search(r'(19|20)(\d+)[\s\S]*生', content)
        if result:
            birthdates.append(result.group(1)+result.group(2))
        else:
            birthdates.append('Unknown')
        
        iter.set_description_str(f'{singer_name}[{birthdates[-1]}]')
    
    # 关闭WebDriver
    driver.quit()
    
    # 将结果保存到DataFrame
    df = pd.DataFrame({'Singer': singer_list, 'Birthdate': birthdates})
    df.to_csv('birthdates.csv', index=False, encoding='utf_8_sig')
    
    print("出生日期提取完成，并保存到 birthdates.csv 文件。")


if __name__ == '__main__':
    
    # # 处理原始数据，提取歌手名和歌词
    # singers = Singers()
    # for i in range(1, 6):
    #     with open(f'lyrics{i}.json', 'r', encoding='utf-8') as f:
    #         data = json.load(f)
            
    #         for item in data:
    #             song_name = item['name']
    #             singer_name = item['singer']
    #             liric = item['lyric']
                
    #             if singer_name not in singers.singers_name:
    #                 print('\nAdd new singer: ', singer_name)
    #                 new_singer = Singer(singer_name)
    #                 new_singer.add_song(Song(song_name, liric))
    #                 singers.add_singer(new_singer)
    #             else:
    #                 print(f'\rSinger {singer_name} adds new song: {song_name}'.ljust(100), end='')
    #                 singers.add_song(singer_name, Song(song_name, liric))

    # # 按歌曲数量过滤歌手
    # singers.filter_by_songs_num(5)
    # singers.save()
    
    # # 爬取歌手的出生日期
    # df = pd.read_csv('./singer_name.csv')
    # singers_list = df['singer_name'].values.tolist()
    # get_birthdates(singers_list)
    
    # 合并歌手信息和出生日期
    df1 = pd.read_csv('./filtered.csv')
    df2 = pd.read_csv('./birthdates.csv')
    df = pd.merge(df1, df2, left_on='singer_name', right_on='Singer')
    # 生日为Unknown的歌手不要
    df = df[df['Birthdate'] != 'Unknown']
    # 生日不在1960~2000年之间的歌手不要
    df = df[(df['Birthdate'] >= '1960') & (df['Birthdate'] <= '2000')]
    # 生日加25年，然后再加-5~5年的随机数，作为歌的年份
    df['Year'] = df['Birthdate'].apply(lambda x: int(x)+30+np.random.randint(-5, 6))
    # lyric列，从字符串列表合并成一个字符串
    df['lyric'] = df['lyric'].apply(lambda x: ''.join(eval(x)))
    df.to_csv('data1.csv', index=False, encoding='utf_8_sig')
    print('最终处理完还剩下', len(df), '条数据。')