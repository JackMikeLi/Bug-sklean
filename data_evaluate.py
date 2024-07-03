'''
真实值与预测值进行绘图分析
1个图有两个子图：
上面的是用one-hot编码数据的预测结果
下面的是用手动编码数据的预测结果
将绘制的图片保存在./img文件夹当中
'''
# 导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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


def get_data(filename):
    data_one_hot = pd.read_excel(filename[0], sheet_name='predict')
    data_hand_code = pd.read_excel(filename[1], sheet_name='predict')
    data_one_hot = data_one_hot.drop('Unnamed: 0', axis=1)
    data_hand_code = data_hand_code.drop('Unnamed: 0', axis=1)

    # data = pd.read_excel(filename,sheet_name='asses')
    # asses_data.rename(columns={'Unnamed: 0':'模型'},inplace=True)

    model_name = data_hand_code.columns
    model_name = model_name.drop(model_name[0]).values

    return [data_one_hot, data_hand_code], model_name

def read_data():#读数据
    hand_code = 'hand-code'
    one_hot_code = 'one-hot'
    code = one_hot_code
    filename = ['one-hot-predict.xlsx', 'hand-code-predict.xlsx']
    pred_data, modelname = get_data(filename)
    return pred_data,modelname


def plot_diffences(pred_data, model_name):#绘图
    x_1 = pred_data[0].index.values
    x_2 = pred_data[1].index.values
    true_1 = pred_data[0]['真实值']
    true_2 = pred_data[1]['真实值']
    for i in model_name:
        fig = plt.figure(figsize=(10, 8))
        model_1 = pred_data[0]
        model_2 = pred_data[1]
        title = 'True Vs ' + i
        plt.title(title)
        plt.rcParams['font.sans-serif'] = 'FangSong'  # 设置显示中文
        # 绘图
        # 子图1
        axes1 = plt.subplot(211)
        axes1.plot(x_1, true_1, 'r')
        axes1.plot(x_2, model_1[i], 'b')

        axes1.set_xlabel('数据索引')
        axes1.set_ylabel('房价')
        axes1.legend(['True', str(i)], fontsize=10, loc='upper right')

        # 子图2
        axes2 = plt.subplot(212, sharex=axes1, sharey=axes1)
        axes2.plot(x_2, true_2, 'r')
        axes2.plot(x_2, model_2[i], 'b')

        axes2.set_xlabel('数据索引')
        axes2.set_ylabel('房价')
        axes2.legend(['True', str(i)], fontsize=10, loc='upper right')

        # 去掉默认的x,y轴范围（即刻度标签）
        axes1.set_xticks([])
        axes1.set_yticks([])
        axes2.set_xticks([])
        axes2.set_yticks([])
        fig.tight_layout()  # 自动调整布局
        fig.savefig(
            fname='./plot/' + title,
            dpi=1000,
        )
def start():
    #读取数据
    pred_data,modelname = read_data()
    #开始绘图
    plot_diffences(pred_data, modelname)
# start()
