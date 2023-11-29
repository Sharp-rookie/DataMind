import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']


birth = pd.read_csv('birthdates.csv')

# # 删除Birthdate是Unknown的
# birth = birth[birth['Birthdate'] != 'Unknown']
# # 过滤掉出生日期在1960年之前的
# birth = birth[birth['Birthdate'] >= '1960']
# birth = birth[birth['Birthdate'] <= '2000']

# # 按每10年聚类到一起并可视化
# birth['Year'] = birth['Birthdate']
# birth = birth.groupby('Year').count()
# birth = birth.reset_index()
# birth['Year'] = birth['Year'].apply(lambda x: x[:3] + '0s')
# birth = birth.groupby('Year').sum()

# # 画图
# plt.figure(figsize=(5, 4))
# plt.bar(birth.index, birth['Singer'])
# plt.xticks(rotation=45)
# plt.xlabel('Year')
# plt.ylabel('Number of Singers')
# plt.title('Number of Singers Born in Each Decade')
# plt.savefig('visual/birth.png', dpi=300, bbox_inches='tight')


data1 = pd.read_csv('data1.csv')
# data1 = data1[data1['Year'] >= 1970]
# data1 = data1[data1['Year'] <= 2023]

# # 按每10年聚类到一起并可视化，这里Year是int
# data1['Year'] = data1['Year'].apply(lambda x: str(x)[:3] + '0s')
# data1 = data1.groupby('Year').count()
# data1 = data1.reset_index()

# # 画图
# plt.figure(figsize=(5, 4))
# plt.bar(data1['Year'], data1['song_name'])
# plt.xticks(rotation=45)
# plt.xlabel('Year')
# plt.ylabel('Number of Songs')
# plt.title('Number of Songs in Each Decade')
# plt.savefig('visual/song.png', dpi=300, bbox_inches='tight')

print(f'歌曲总数：{len(list(data1["song_name"]))}, 歌手总数：{(len(list(set(birth["Singer"]))))}')


song_num = []
singer_num = []
for theme in ['爱情', '成长', '教育', '励志', '红色']:
    data = pd.read_csv(f'主题/{theme}.csv')
    
    # column = song,singer,lirics
    song_num.append(len(list(data['song'])))
    singer_num.append(len(list(set(data['singer']))))
    
# 画图
plt.figure(figsize=(5, 4))
plt.bar(['爱情', '成长', '教育', '励志', '红色'], song_num)
plt.xticks(rotation=45)
plt.xlabel('Theme')
plt.ylabel('Number of Songs')
plt.title('Number of Songs in Each Theme')
plt.savefig('visual/theme.png', dpi=300, bbox_inches='tight')

print(f'歌曲总数：{sum(song_num)}, 歌手总数：{sum(singer_num)}')