#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:data_handle.py
@TIME:2019/8/19 14:43
@DES:
'''


import  pandas as pd
from keras.utils import np_utils
from pre_handle import *


# 获取chars的pd列表
'''
输入：字符集文件【chars08.txt】
输出：pd格式的chars
'''
def get_chars(chars_file):
    '''
    :param chars_file: 字典文件
    :return:  规范了数据格式的字典
    '''
    # 获取chars
    chars = open(chars_file).read().decode('utf-8')  #chars必须要是unnicode
    # chars = open(chars_file).read()
    chars_list = list(chars)
    # chars = pd.Series(chars_list)
    chars = pd.Series(chars_list).value_counts()
    # 按道理说是可以不用这一步的
    chars[:] = range(1, len(chars) + 1)
    return chars


# 加载数据
'''
输入：指定的【经过】tag处理的文件、完成的字符集【chars08.txt】、【maxlan】
返回：dataFrame格式的数据
'''
def init_datas(data_file,chars,maxlen):
    '''
    :param data_file: 标注了词位的语料文件
    :param chars: 语料文件对应的字典
    :param maxlen: 神经网络的输入长度
    :return: 符合神经网络输入规格的输入数据
    '''
    s = open(data_file).read().decode('utf-8')
    # chars_in = codecs.open(chars_file, 'r', 'utf-8')
    s = s.split('\r\n')
    s = u''.join(map(clean, s))
    s = re.split(u'[，。！？、]/[bems]', s)
    data = []  # 生成训练样本
    label = []  #生成训练标签
    for i in s:
        x = get_xy(i)
        if x:
            data.append(x[0])
            label.append(x[1])

    d = pd.DataFrame(index=range(len(data)))
    d['data'] = data
    d['label'] = label
    d = d[d['data'].apply(len) <= maxlen]
    d.index = range(len(d))
    tag = pd.Series({u's': 0, u'b': 1, u'm': 2, u'e': 3, u'x': 4})

    d['x'] = d['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))

    def trans_one(x):
        _ = map(lambda y: np_utils.to_categorical(y, 5), tag[x].reshape((-1, 1)))
        _ = list(_)
        _.extend([np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x)))
        return np.array(_)

    d['y'] = d['label'].apply(trans_one)
    return d


    # return data


if __name__ =="__main__":
    chars = get_chars("dictionary/chars02_pku.txt")
    data = init_datas("data_set/test_pku.txt",chars,32)
    print("执行完毕！")


