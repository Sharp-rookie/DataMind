import os
import re
import pandas as pd


df = pd.read_excel('./theme.xlsx', sheet_name=['爱情', '红色', '成长', '教育', '励志'])

for sheet_name in ['爱情', '成长', '教育', '励志', '红色']:
    # 切换sheet
    sheet = df.get(sheet_name)
    
    filtered_row = []
    for index, row in sheet.iterrows():
        song = row['歌曲名']
        singer = row['歌手']
        lirics = row['歌词']
        
        # 正则匹配过滤歌词
        if sheet_name == '红色':
            filtered = re.findall(r"[0-9a-zA-Z'.(/)~@?,一-龥，。【】！’～（）？　 …  ]{10,}", lirics)
        else:
            filtered = re.findall(r"[0-9a-zA-Z'.(/)~@?,一-龥，。【】！’～（）？　   ]{50,}", lirics)
        if filtered:
            # 保留最长的歌词
            clean_lirics = max(filtered, key=lambda x: len(x))
            filtered_row.append([song, singer, clean_lirics])
        else:
            # print(sheet_name, song)
            print('匹配失败：', sheet_name, '《', song, '》', lirics)
        
    # 保存到csv
    result = pd.DataFrame(filtered_row, columns=['song', 'singer', 'lirics'])
    os.makedirs('主题', exist_ok=True)
    result.to_csv(f'主题/{sheet_name}.csv', index=False, encoding='utf_8_sig')