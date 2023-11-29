import re
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd

def get_birthdates(singer_list):
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
    df.to_csv('birthdates.csv', index=False)
    
    print("出生日期提取完成，并保存到 birthdates.csv 文件。")

# 测试函数
df = pd.read_csv('./singer_name.csv')
singers_list = df['singer_name'].values.tolist()
get_birthdates(singers_list)