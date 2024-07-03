'''
主函数
一键完成数据爬取、数据预处理、模型的训练与预测、模型评估
'''
import data_evaluate
import get_data
import data_processed
import data_predict
import data_evaluate
if __name__ == '__main__':
    #数据爬取
    get_data.start()
    #数据预处理
    data_processed.start()
    #模型训练、预测
    data_predict.predict()
    #模型评估
    data_evaluate.start()