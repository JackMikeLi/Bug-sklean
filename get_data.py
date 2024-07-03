'''
爬取链家二手房房源信息
以上海的二手房源为例子
爬取的信息列有：室厅、面积、朝向、装修情况、楼层、年份、房屋类型、别墅类型、总价网址
爬取后的信息存储到house_Data.xlsx当中
'''
import numpy as np
import requests
import os
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def get_url():
    url = []
    for i in range(1, 101):
        u = 'https://sh.lianjia.com/ershoufang/pg{}/'.format(i)
        url.append(u)
    return url


def get_data(url):
    Headers = {
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
        "X-Requested-With": 'XMLHttpRequest'
    }
    response = requests.get(url, headers=Headers)
    response.encoding = 'utf8'
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    div_price = soup.find_all('div', attrs={'class': 'totalPrice totalPrice2'})
    totalprice = []
    for price in div_price:
        # totalprice.append(price.text.replace('万',''))
        totalprice.append(price.text)

    div_address = soup.find_all('div', attrs={'class': 'houseInfo'})
    totaladdress = []
    for address in div_address:
        totaladdress.append(address.text.split('|'))
    return totalprice, totaladdress, [url] * len(totalprice)


def get_all_data():
    df = pd.DataFrame()
    totalprice, totaladdress, html = [], [], []
    for url in tqdm(get_url()):
        price, address, last = get_data(url)
        totalprice.append(price), totaladdress.append(address), html.append(last)
    return totaladdress, totalprice, html


def split_address(data):
    ans = []
    for i in data:
        for j in i:
            ans.append(j)
    return ans


def fill_data(data):
    return data + [0]*(8-len(data))
def fresh_address(data):
    ans = []
    #输入的data是address
    data = split_address(data)
    for i in data:
        if len(i)<8:
            tem = fill_data(i)
            ans.append(tem)
        elif(len(i) == 8):
            ans.append(i)
    return ans
def start():
    # 爬取1-100页数据
    address, price, html = get_all_data()
    columns = ['室厅', '面积', '朝向', '装修情况', '楼层', '年份', '房屋类型', '别墅类型', '总价', '网址']
    res = fresh_address(address)
    res_address = np.array(res).reshape(-1, 8)
    price, html = np.array(price).reshape(-1, 1), np.array(html).reshape(-1, 1)
    final_data = np.concatenate((res_address, price, html), axis=1)
    df = pd.DataFrame(final_data, columns=columns)
    # 定义文件路径（可以使用绝对路径）
    file_path = os.path.join(os.getcwd(), 'house_Data.xlsx')
    # 写入 Excel 文件
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='House', index=True)

    # 确认文件生成的位置
    print(f"文件已保存到: {file_path}")
# start()