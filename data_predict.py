'''
将数据清洗后的数据输入到回归模型当中去
对上海二手房房源总价进行预测
其中主要对比的有两个层面：
（1）9种机器学习回归模型的优劣
（2）手动编码和one-hot编码的优劣
最终数据存储在
hand-code-predict.xlsx,one-hot-data.xlsx
'''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression#线性回归
from sklearn.linear_model import Ridge#岭回归
from sklearn.linear_model import Lasso#Lasso回归
from sklearn.linear_model import ElasticNet#enet回归
from sklearn.tree import DecisionTreeRegressor#决策树
from sklearn.ensemble import RandomForestRegressor#随机森林
from sklearn.svm import SVR#支持向量机
from sklearn.neural_network import MLPRegressor#神经网络回归
from sklearn.linear_model import BayesianRidge#基于贝叶斯的线性回归

def assess(y_test, y_pred):
    # 评估回归模型的预测结果
    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    # 计算均方根误差
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    from sklearn.metrics import mean_absolute_error

    # 计算平均绝对误差
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")

    from sklearn.metrics import r2_score

    # 计算决定系数
    r2 = r2_score(y_test, y_pred)
    print(f"Coefficient of Determination (R^2): {r2}")

    return mse, rmse, mae, r2

def train_predict_model(code,x_train, y_train, x_test, y_test):
    # 模型训练
    # 线性回归
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    # 岭回归
    ridge = Ridge()
    ridge.fit(x_train, y_train)
    # Lasso回归
    lasso = Lasso()
    lasso.fit(x_train, y_train)
    # enet回归
    enet = ElasticNet()
    enet.fit(x_train, y_train)
    # 决策树
    tree = DecisionTreeRegressor()
    tree.fit(x_train, y_train)
    # 随机森林
    randomforest = RandomForestRegressor()
    randomforest.fit(x_train, y_train)
    # 支持向量机
    svr = SVR()
    svr.fit(x_train, y_train)
    # 神经网络回归
    mlpr = MLPRegressor()
    mlpr.fit(x_train, y_train)
    # 基于贝叶斯的线性回归
    bayes = BayesianRidge()
    bayes.fit(x_train, y_train)

    y_test = np.array(y_test).reshape(-1, 1)
    # 模型预测
    y_pred_lr = lr.predict(x_test).reshape(-1, 1)
    y_pred_ridge = ridge.predict(x_test).reshape(-1, 1)
    y_pred_lasso = lasso.predict(x_test).reshape(-1, 1)
    y_pred_enet = enet.predict(x_test).reshape(-1, 1)
    y_pred_tree = tree.predict(x_test).reshape(-1, 1)
    y_pred_randomforest = randomforest.predict(x_test).reshape(-1, 1)
    y_pred_svr = svr.predict(x_test).reshape(-1, 1)
    y_pred_mlpr = mlpr.predict(x_test).reshape(-1, 1)
    y_pred_bayes = bayes.predict(x_test).reshape(-1, 1)

    data = np.concatenate((y_test, y_pred_lr, y_pred_ridge, y_pred_lasso, y_pred_enet, y_pred_tree, y_pred_randomforest,
                           y_pred_svr, y_pred_mlpr, y_pred_bayes), axis=1)
    columns = ['真实值', 'lr', 'ridge', 'lasso', 'enet', 'tree', 'randomforest', 'svr', 'mlpr', 'bayes']
    df1 = pd.DataFrame(data, columns=columns)
    # 计算指标
    data2 = []
    for i in df1.columns:
        data2.append(assess(y_test, df1[i]))
    df2 = pd.DataFrame(data2, index=columns, columns=['mse', 'rmse', 'mae', 'r2'])
    # 写入到文件中
    file_path = os.path.join(os.getcwd(), code + '-predict.xlsx')
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='predict')
        df2.to_excel(writer, sheet_name='asses')
    return df1, df2

def select_data(data,last_index):
    # 计算皮尔逊相关系数和斯皮尔曼相关系数
    pearson_corr = data.corr(method='pearson')[last_index]
    spearman_corr = data.corr(method='spearman')[last_index]
    # 设定阈值
    threshold = 0.1

    # 筛选皮尔逊相关性高于阈值的特征
    selected_features_pearson = pearson_corr[abs(pearson_corr) > threshold].index.values
    selected_features_pearson = selected_features_pearson[selected_features_pearson != last_index]

    # 筛选斯皮尔曼相关性高于阈值的特征
    selected_features_spearman = spearman_corr[abs(spearman_corr) > threshold].index.values
    selected_features_spearman = selected_features_spearman[selected_features_spearman != last_index]

    # 合并两种方法选择的特征列
    selected_features = np.concatenate((selected_features_pearson, selected_features_spearman)).astype(str)
    selected_features = np.unique(selected_features).astype(int)
    # 提取相关性较高的特征列
    selected_data = data[selected_features]

    print("皮尔逊选择：:", selected_features_pearson)
    print("斯皮尔曼选择:", selected_features_spearman)
    print("最终选择的特征列（剔除后合并）:", selected_features)
    print(spearman_corr, pearson_corr)
    return selected_data

def predict():
    code = ['hand-code','one-hot']
    for c in code:
        print(c)
        #读取数据
        filename = c + '-data.xlsx'
        data = pd.read_excel(filename, skiprows=1, header=None)
        data = data.drop(data.columns[0], axis=1)
        last_index = len(data.loc[1])

        #特征提取
        x = select_data(data,last_index)
        y = data[last_index]
        x.columns = x.columns.astype(str)
        y.columns = x.columns.astype(str)

        #划分训练集、测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        #开始训练
        df1, df2 = train_predict_model(c,x_train, y_train, x_test, y_test)
        print(df1,df2)
# predict()
