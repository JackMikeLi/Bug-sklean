'''
对爬取到的数据进行数据预处理
数据预处理包括：
数据清洗、特征编码、特征选择
最终得到两个数据
hand-code-data.xlsx:手动编码后的特征数据
one-hot-data.xlsx:使用one-hot编码后的特征数据
两个数据可输入到机器学习模型当中进行预测
'''
import numpy as np
import re
import pandas as pd
def read_data():#读取数据
    data = pd.read_excel('house_Data.xlsx')
    data = data.drop('Unnamed: 0', axis=1)
    return data

def fresh_data(data):#清洗数据
    # 剔除数据中的所有空格
    for index, row in data.iterrows():
        for i, j in enumerate(row):
            row[i] = j.replace(' ', '')
        data.loc[index] = row

    house = []
    for index, row in data.iterrows():
        # 更正数据
        if row['年份'] == '暂无数据':
            row['年份'] = 0
        if row['年份'] == '板楼':
            row['房屋类型'] = row['年份']
            row['年份'] = 0
        if row['年份'] == '板塔结合':
            row['房屋类型'] = row['年份']
            row['年份'] = 0
        if row['房屋类型'] == '暂无数据':
            row['房屋类型'] = 0
        # 剔除多余数据
        if '车位' not in row['室厅']:
            house.append(row)
    house = pd.DataFrame(house)
    # 检查是否有缺失值,是否有重复值
    print('缺失值情况:{},重复值情况:{}'.format(house.isna().sum(), house.duplicated().sum()))
    return house

def parse_floor_info(floor_str):# 分离楼层类型和总楼层数的函数
    floor_type_match = re.search(r'(低楼层|中楼层|高楼层)', floor_str)
    total_floors_match = re.search(r'共(\d+)层', floor_str)

    if floor_type_match:
        floor_type = floor_type_match.group(1)
        floor_type = floor_type.split('楼')[0]
    else:
        floor_type = '未知'  # 如果没有匹配到楼层类型，设为未知

    if total_floors_match:
        total_floors = int(total_floors_match.group(1))
    else:
        total_floors = int(floor_str.split('层')[0])  # 处理只有总楼层数的情况
    return floor_type, total_floors

def get_exact_data(house):#提取数据
    # 提取数据
    room_1, room_2, squre, direction = [], [], [], []
    res, y = [], []  # res为楼层信息
    for index, row in house.iterrows():
        # 提取room
        ans = row['室厅'].replace('室', ',')
        ans = ans.replace('厅', '')
        a, b = ans.split(',')
        room_1.append(int(a))
        room_2.append(int(b))

        # 提取面积数据
        ans = row['面积'].replace('平米', '')
        squre.append(float(ans.split()[0]))

        # 对楼层数据进行处理
        ans = parse_floor_info(row['楼层'])
        if ('未知' in ans[0]):
            ans = ans[1]
        else:
            ans = ans[0]
        res.append(ans)

        # 对总价y进行处理
        y.append(float(row['总价'].replace('万', '')))

    n = np.concatenate((np.array(room_1).reshape(-1,1),np.array(np.array(room_2)).reshape(-1,1),np.array(np.array(squre)).reshape(-1,1),
                        np.array(np.array(house['朝向'])).reshape(-1,1),np.array(np.array(house['装修情况'])).reshape(-1,1),np.array(np.array(res)).reshape(-1,1),
                        np.array(np.array(house['年份'])).reshape(-1,1),np.array(np.array(house['房屋类型'])).reshape(-1,1),np.array(np.array(house['别墅类型'])).reshape(-1,1),
                        np.array(np.array(y)).reshape(-1,1)),axis=1)
    df= pd.DataFrame(n, columns=['室', '厅', '面积', '朝向', '装修情况', '楼层','年份', '房屋类型', '别墅类型', '总价'])
    return df

def hand_code(df):#手动编码
    # 朝向编码
    df_hand = df.copy()
    dir = set(list(df_hand['朝向']))
    count = 0
    dirDictF, dirDictS = {}, {}
    for i in dir:
        dirDictF[i] = count
        dirDictS[count] = i
        count += 1
    for index, row in df_hand.iterrows():
        row['朝向'] = dirDictF[row['朝向']]

    # 装修编码
    type = set(list(df_hand['装修情况']))
    count = 0
    typeDictF, typeDictS = {}, {}
    for i in type:
        typeDictF[i] = count
        typeDictS[count] = i
        count += 1
    for index, row in df_hand.iterrows():
        row['装修情况'] = typeDictF[row['装修情况']]

    # 楼层编码
    height = set(list(df_hand['楼层']))
    count = 0
    heightDictF, heightDictS = {}, {}
    for i in height:
        heightDictF[i] = count
        heightDictS[count] = i
        count += 1
    for index, row in df_hand.iterrows():
        row['楼层'] = heightDictF[row['楼层']]

    # 年份编码
    year = set(list(df_hand['年份']))
    count = 0
    yearDictF, yearDictS = {}, {}
    for i in year:
        yearDictF[i] = count
        yearDictS[count] = i
        count += 1
    for index, row in df_hand.iterrows():
        row['年份'] = yearDictF[row['年份']]

    # 房屋类型编码
    housetype = set(list(df_hand['房屋类型']))
    count = 0
    housetypeDictF, housetypeDictS = {}, {}
    for i in housetype:
        housetypeDictF[i] = count
        housetypeDictS[count] = i
        count += 1
    for index, row in df_hand.iterrows():
        row['房屋类型'] = housetypeDictF[row['房屋类型']]

    # 别墅类型编码
    houseup = set(list(df_hand['别墅类型']))
    count = 0
    houseupDictF, houseupDictS = {}, {}
    for i in houseup:
        houseupDictF[i] = count
        houseupDictS[count] = i
        count += 1
    for index, row in df_hand.iterrows():
        row['别墅类型'] = houseupDictF[row['别墅类型']]
    df_hand.to_excel('hand-code-data.xlsx')
    # 确认文件生成的位置
    print(f"手动编码文件已保存到: {'hand-code-data.xlsx'}")
    print(df)
    return df

def one_hot_code(df,length):
    # one-hot编码
    df_one_hot = df.copy()
    x1 = pd.get_dummies(df_one_hot['室'], columns='室').astype(int).values
    x1_columns = pd.get_dummies(df_one_hot['室'], columns='室').astype(int).columns.values
    x2 = pd.get_dummies(df_one_hot['厅'], columns='厅').astype(int).values
    x2_columns = pd.get_dummies(df_one_hot['厅'], columns='厅').astype(int).columns.values
    x3 = np.array(df_one_hot['面积'])
    x3_columns = np.array(['面积'])
    x4 = pd.get_dummies(df_one_hot['朝向'], columns='朝向').astype(int).values
    x4_columns = pd.get_dummies(df_one_hot['朝向'], columns='朝向').astype(int).columns.values
    x5 = pd.get_dummies(df_one_hot['装修情况'], columns='装修').astype(int).values
    x5_columns = pd.get_dummies(df_one_hot['装修情况'], columns='装修').astype(int).columns.values
    x6 = pd.get_dummies(df_one_hot['装修情况'], columns='装修').astype(int).values
    x6_columns = pd.get_dummies(df_one_hot['装修情况'], columns='装修').astype(int).columns.values

    x7 = pd.get_dummies(df_one_hot['楼层'], columns='楼层').astype(int).values
    x7_columns = pd.get_dummies(df_one_hot['楼层'], columns='楼层').astype(int).columns.values
    x8 = pd.get_dummies(df_one_hot['年份'], columns='年份').astype(int).values
    x8_columns = pd.get_dummies(df_one_hot['年份'], columns='年份').astype(int).columns.values

    x9 = pd.get_dummies(df_one_hot['房屋类型'], columns='房屋类型').astype(int).values
    x9_columns = pd.get_dummies(df_one_hot['房屋类型'], columns='房屋类型').astype(int).columns.values
    x10 = pd.get_dummies(df_one_hot['别墅类型'], columns='别墅类型').astype(int).values
    x10_columns = pd.get_dummies(df_one_hot['别墅类型'], columns='别墅类型').astype(int).columns.values
    y = pd.DataFrame(df_one_hot['总价'], columns=['总价'])
    # 组合列名
    columns = np.concatenate((x1_columns, x2_columns, x3_columns, x4_columns, x5_columns, x6_columns, x7_columns,
                              x8_columns, x9_columns, x10_columns))

    # 组合特征变量
    x1 = x1.reshape(length, -1)
    x2 = x2.reshape(length, -1)
    x3 = x3.reshape(-1, 1)
    x4.reshape(length, -1)
    x5.reshape(length, -1)
    x6.reshape(length, -1)
    x7.reshape(length, -1)
    x8.reshape(length, -1)
    x9.reshape(length, -1)
    x10.reshape(length, -1)
    np.ndim(x1), np.ndim(x2), np.ndim(x3), np.ndim(x4), np.ndim(x5), np.ndim(x6), np.ndim(x7), np.ndim(x8)
    x = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), axis=1)
    x = pd.DataFrame(x, columns=columns)
    df_one_hot = pd.concat([x, y], axis=1)
    df_one_hot.to_excel('one-hot-data.xlsx')
    # 确认文件生成的位置
    print(f"one-hot编码文件已保存到: {'hand-code-data.xlsx'}")
    print(df)
    return df
def start():
    #读取数据
    data = read_data()
    #清洗数据
    data = fresh_data(data)
    length = len(data)
    #数据提取
    data = get_exact_data(data)
    print(data)
    #手动编码
    df_hand = hand_code(data)
    #one-hot编码
    df_one_hot = one_hot_code(data,length)
    print(df_hand)
    print(df_one_hot)
#start()






