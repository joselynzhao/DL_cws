#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:model_evaluate.py
@TIME:2019/8/19 14:45
@DES:
'''


import  re
import numpy as np
import  pandas as pd
from keras.utils import np_utils

from keras import  Sequential
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,Dropout
from keras.models import Model
from keras.models import load_model

from pre_handle import *
from data_handle import *
from model_train import *


# 评估模型
'''
dec：返回和输出评估结果
'''
def evaluate_model(model,test_data,maxlen,batch_size,cost_file):
    print("开始评估！")
    loss_and_accuracy = model.evaluate(np.array(list(test_data['x'])),
                                       np.array(list(test_data['y'])).reshape((-1, maxlen, 5)),
                                       batch_size=batch_size)
    print loss_and_accuracy
    # 把评估结果写到 cost file里面去
    cost_f = codecs.open(cost_file, 'a', 'utf-8')
    cost_f.write(str(loss_and_accuracy[0])+'\n')
    cost_f.write(str(loss_and_accuracy[1])+'\n')
    cost_f.close()




if __name__ =="__main__":
    #超参数设置
    maxlen = 32
    batch_size = 64 #不指定则默认为32

    #加载测试数据集和字典
    chars_file = "dictionary/chars02_pku.txt"
    test_file = "data_set/test_pku.txt"
    chars = get_chars(chars_file)
    test_data = init_datas(test_file, chars, maxlen)

    # 加载模型
    model_name = "model_pku_lstm"
    model_file = "model_save/" + model_name + ".h5"
    model = get_model(model_file)

    # 评估模型
    cost_file = "cost_file/"+model_name+".txt"
    evaluate_model(model,test_data,maxlen,batch_size,cost_file)
