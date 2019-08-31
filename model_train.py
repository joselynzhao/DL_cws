#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:model_train.py
@TIME:2019/8/19 14:44
@DES:
'''


from keras import  Sequential
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,Dropout

from keras.models import load_model


from data_handle import *
from pre_handle import *

from keras.callbacks import TensorBoard

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)



# 生成模型
'''
输入：【maxlen】、完成字符集的长度【len_chars】、字向量长度【word_size】、需要保持模型的文件名称
返回：生成的模型
'''

def gener_model01(maxlen,len_chars,word_size,num_lstm,model_file_name):
    '''
    :param maxlen: 输入长度
    :param len_chars:  字典长度
    :param word_size: 字向量长度
    :param num_lstm: lstm节点数
    :param model_file_name: 模型的保持文件
    :return:
    '''
    model = Sequential()
    model.add(Embedding(len_chars + 1, word_size, input_length=maxlen))
    '''len_chars+1是输入维度，word_size是输出维度，input_length是节点数'''
    model.add(LSTM(num_lstm, return_sequences=True))
    model.add(LSTM(num_lstm, return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(Dense(5,activation='softmax'))
    model.add(TimeDistributed(Dense(5, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的权值，每层输出值的分布直方图
    model.save(model_file_name)
    model.summary()
    return model



def gener_model02(maxlen,len_chars,word_size,num_lstm,model_file_name):
    model = Sequential()
    model.add(Embedding(len_chars + 1, word_size, input_length=maxlen))
    '''len_chars+1是输入维度，word_size是输出维度，input_length是节点数'''
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True), merge_mode='sum'))
    # model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True), merge_mode='sum'))
    model.add(TimeDistributed(Dense(5, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(model_file_name)
    model.summary()
    return model


def gener_model03(maxlen,len_chars,word_size,num_lstm,model_file_name):
    model = Sequential()
    model.add(Embedding(len_chars + 1, word_size, input_length=maxlen))
    '''len_chars+1是输入维度，word_size是输出维度，input_length是节点数'''
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True), merge_mode='sum'))
    # model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True), merge_mode='sum'))
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True), merge_mode='sum'))
    model.add(TimeDistributed(Dense(5, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(model_file_name)
    model.summary()
    return model


# 加载模型
'''
输入：指定需要加载的model的文件
返回：model
'''
def get_model(model_file_name):
    try:
        import h5py

        # print ('import fine')
    except ImportError:
        h5py = None

    model = load_model(model_file_name)
    print str(model_file_name)+" 加载成功~"
    model.summary()
    return model


# 训练模型

'''
dec：采用自我评估的方式,validation_split=0.1
'''
def train_model02(model,train_data,maxlen,batch_size,epoch,model_file,cost_file,split):
    cost_f = codecs.open(cost_file, 'a', 'utf-8')
    history = model.fit(np.array(list(train_data['x'])), np.array(list(train_data['y'])).reshape((-1, maxlen, 5)),
                        batch_size=batch_size, validation_split=split,epochs=epoch)
    # print(history) callbacks=[tbCallBack]
    s = str(history.history)
    # print s
    s = re.findall('\[(.*?)\]', s)
    tra_acc = s[0]
    tra_loss = s[1]
    val_acc = s[2]
    val_loss = s[3]
    print("分别输出tra_acc,tra_loss,val_acc,val_loss")
    print tra_acc
    print tra_loss
    print val_acc
    print val_loss
    cost_f.write(str(tra_acc)+' \n')
    cost_f.write(str(tra_loss)+' \n')
    cost_f.write(str(val_acc)+' \n')
    cost_f.write(str(val_loss)+' \n')
    cost_f.close()
    model.save(model_file)





# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_filepath = '/tmp/keras_log'

if __name__ == "__main__":
    # 超参数设定
    maxlen = 32
    epoch = 5
    batch_size = 512
    word_size =128
    num_lstm = 128

    # 加载训练数据集和字典
    chars_file = "dictionary/chars02_msr.txt"
    train_file = "data_set/train_msr.txt"
    chars = get_chars(chars_file)
    train_data = init_datas(train_file, chars, maxlen)

    # 搭建模型
    model_name = "model_msr_final"
    model_file ="model_save/"+model_name+".h5"
    # model = gener_model02(maxlen,len(chars),word_size,num_lstm,model_file)
    model = get_model(model_file)

    # 开始训练
    cost_file= "cost_file/"+model_name+".txt"
    split = 0.1
    train_model02(model,train_data,maxlen,batch_size,epoch,model_file,cost_file,split)




