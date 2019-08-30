#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWARE:PyCharm
@FILE:tool.py
@TIME:2019/8/22 上午9:23
@DES:  用于绘制图像
'''
import matplotlib.pyplot as plt
import macpath
import numpy as np
import  codecs

import  re
def draw_cost_file(cost_file):
    cost_f = codecs.open(cost_file, 'r', 'utf-8')
    datas = []
    for line in cost_f.readlines():
        data_list = line.strip().split(', ')
        data_list_float = []
        for data in data_list:
            data = float(data)
            data_list_float.append(data)
        print(data_list_float)
        datas.append(data_list_float)
    tra_acc = datas[0]
    tra_loss = datas[1]
    val_acc = datas[2]
    val_loss = datas[3]
    all_len = len(tra_acc)
    # eva_loss = datas[4][0]
    # eva_acc = datas[5][0]
    plt.figure()

    plt.subplot(2,1,1)
    # print len(tra_acc[10:])

    # tra_acc=tra_acc[5:]
    # val_acc=val_acc[5:]
    x1 = np.linspace(5, all_len, all_len-5)
    max_tra_acc = np.argmax(tra_acc)  # max value index
    max_val_acc = np.argmax(val_acc)  # min value index
    # min_indx = np.argmin(a)  # min value index
    plt.plot(x1,tra_acc[5:],'r',label='tra_acc')
    plt.plot(x1,val_acc[5:],"g",label='val_acc')
    plt.plot(max_tra_acc, tra_acc[max_tra_acc], 'rs')
    plt.plot(max_val_acc, val_acc[max_val_acc], 'gs')
    # show_max = str(tra_acc[max_tra])
    # show_max = '[' + str(max_tra) + ' ' + str(tra_acc[max_tra]) + ']'
    plt.annotate(str(tra_acc[max_tra_acc])[:6], xy=(max_tra_acc, tra_acc[max_tra_acc]))
    plt.annotate(str(val_acc[max_val_acc])[:6], xy=(max_val_acc, val_acc[max_val_acc]))
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("tra_acc vs val_acc")
    plt.legend(loc='upper left')
    # plt.imshow()


    plt.subplot(2, 1, 2)
    # min_indx = np.argmin(a)  # min value index

    # tra_loss = tra_loss[10:]
    # val_loss = val_loss[10:]
    x2 = np.linspace(5, all_len, all_len-5)
    max_tra_loss = np.argmin(tra_loss)  # max value index
    max_val_loss = np.argmin(val_loss)  # min value index
    plt.plot(x2, tra_loss[5:], 'r', label='tra_loss')
    plt.plot(x2, val_loss[5:], "g", label='val_loss')
    plt.plot(max_tra_loss, tra_loss[max_tra_loss], 'rs')
    plt.plot(max_val_loss, val_loss[max_val_loss], 'gs')
    # show_max = str(tra_acc[max_tra])
    # show_max = '[' + str(max_tra) + ' ' + str(tra_acc[max_tra]) + ']'
    plt.annotate(str(tra_loss[max_tra_loss])[:6],  xy=(max_tra_loss, tra_loss[max_tra_loss]))
    plt.annotate(str(val_loss[max_val_loss])[:6],  xy=(max_val_loss, val_loss[max_val_loss]))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("tra_loss vs val_loss")
    plt.legend(loc='lower left')

    plt.show()
    cost_f.close()

def draw_acc(cost_file):
    cost_f = codecs.open(cost_file, 'r', 'utf-8')
    datas = []
    for line in cost_f.readlines():
        data_list = line.strip().split(', ')
        data_list_float = []
        for data in data_list:
            data = float(data)
            data_list_float.append(data)
        print(data_list_float)
        datas.append(data_list_float)
    tra_acc = datas[0]
    val_acc = datas[2]
    all_len = len(tra_acc)
    plt.figure()
    x1 = np.linspace(5, all_len, all_len - 5)
    max_tra_acc = np.argmax(tra_acc)  # max value index
    max_val_acc = np.argmax(val_acc)  # min value index
    # min_indx = np.argmin(a)  # min value index
    plt.plot(x1, tra_acc[5:], 'r', label='tra_acc')
    plt.plot(x1, val_acc[5:], "g", label='val_acc')
    plt.plot(max_tra_acc, tra_acc[max_tra_acc], 'rs')
    plt.plot(max_val_acc, val_acc[max_val_acc], 'gs')
    # show_max = str(tra_acc[max_tra])
    # show_max = '[' + str(max_tra) + ' ' + str(tra_acc[max_tra]) + ']'
    plt.annotate(str(tra_acc[max_tra_acc])[:6], xy=(max_tra_acc, tra_acc[max_tra_acc]))
    plt.annotate(str(val_acc[max_val_acc])[:6], xy=(max_val_acc, val_acc[max_val_acc]))
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("tra_acc vs val_acc")
    plt.legend(loc='upper left')
    plt.show()
    cost_f.close()

def draw_loss(cost_file):
    cost_f = codecs.open(cost_file, 'r', 'utf-8')
    datas = []
    for line in cost_f.readlines():
        data_list = line.strip().split(', ')
        data_list_float = []
        for data in data_list:
            data = float(data)
            data_list_float.append(data)
        print(data_list_float)
        datas.append(data_list_float)

    tra_loss = datas[1]
    val_loss = datas[3]
    all_len = len(tra_loss)
    plt.figure()
    x2 = np.linspace(5, all_len, all_len - 5)
    max_tra_loss = np.argmin(tra_loss)  # max value index
    max_val_loss = np.argmin(val_loss)  # min value index
    plt.plot(x2, tra_loss[5:], 'r', label='tra_loss')
    plt.plot(x2, val_loss[5:], "g", label='val_loss')
    plt.plot(max_tra_loss, tra_loss[max_tra_loss], 'rs')
    plt.plot(max_val_loss, val_loss[max_val_loss], 'gs')
    # show_max = str(tra_acc[max_tra])
    # show_max = '[' + str(max_tra) + ' ' + str(tra_acc[max_tra]) + ']'
    plt.annotate(str(tra_loss[max_tra_loss])[:6], xy=(max_tra_loss, tra_loss[max_tra_loss]))
    plt.annotate(str(val_loss[max_val_loss])[:6], xy=(max_val_loss, val_loss[max_val_loss]))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("tra_loss vs val_loss")
    plt.legend(loc='lower left')

    plt.show()
    cost_f.close()

def draw_compare_acc(cost_file_list,file_name_list,plt_title):
    cost_f_list = []
    for cost_file in cost_file_list:
        cost_f = codecs.open(cost_file,'r','utf-8')
        cost_f_list.append(cost_f)
    cost_datas = []
    for cost_f in cost_f_list:
        datas=[]
        for line in cost_f.readlines():
            data_list = line.strip().split(', ')
            data_list_float = []
            for data in data_list:
                data = float(data)
                data_list_float.append(data)
            # print(data_list_float)
            datas.append(data_list_float)
        cost_datas.append(datas)
    length = len(cost_datas[0][0])


    #开始绘图
    plt.figure()
    colors = ['r','g','b','y','k']
    start_point=5
    x1 = np.linspace(start_point,length,length-start_point)
    print "len x = "+str(len(x1))
    for i in range(len(cost_datas)):
        line = cost_datas[i][2] #2表示测试准确率
        max_point = np.argmax(line)
        plt.plot(x1,line[start_point:],colors[i],label=file_name_list[i])
        plt.plot(max_point,line[max_point],colors[i]+'s')
        plt.annotate(str(line[max_point])[:6],xy=(max_point,line[max_point]))
    plt.xlabel("epoch")
    plt.ylabel("val_acc")
    plt.title(plt_title)
    plt.legend(loc='lower right')
    plt.show()
    for cost_f in cost_f_list:
        cost_f.close()



if __name__ =="__main__":

    cost_file = "cost_file/model_pku_04.txt"
    # draw_acc(cost_file)
    # draw_loss(cost_file)

    pre = "cost_file/model_"

    # 网络层数对比
    # cost_file_list = [pre+"pku_l1.txt",pre+"pku_03.txt",pre+"pku_04.txt"]
    # file_name_list = ["pku_l1","pku_l2","pku_l3"]
    # plt_title = "num_of_layers"

    # lstm vs blstm
    cost_file_list = [pre + "pku_lstm.txt", pre + "pku_03.txt"]
    file_name_list = ["pku_lstm", "pku_blstm"]
    plt_title = "lstm vs blstm"

    # 隐藏层结点数对比
    # cost_file_list = [pre + "pku_n64.txt", pre + "pku_04.txt", pre + "pku_n192.txt",pre+"pku_n256.txt"]
    # file_name_list = ["pku_n64", "pku_n128", "pku_n192","pku_n256"]
    # plt_title = "node_of_lstm"

    # 字向量长度
    # cost_file_list = [pre + "pku_w64.txt", pre + "pku_04.txt", pre + "pku_w192.txt",pre+'pku_w256.txt']
    # file_name_list = ["pku_w64", "pku_w128", "pku_w192","pku_w256"]
    # plt_title = "length_of_word"

    # batch_size
    # cost_file_list = [pre + "pku_b512.txt", pre + "pku_04.txt", pre + "pku_b1536.txt"]
    # file_name_list = ["pku_b512", "pku_b1024", "pku_b1536"]
    # plt_title = "batch_size"

    # 数据集
    # cost_file_list = [pre + "pku_04.txt", pre + "msr.txt", pre + "as.txt"]
    # file_name_list = ["pku", "msr", "as"]
    # plt_title = "data_set"

    draw_compare_acc(cost_file_list,file_name_list,plt_title)

