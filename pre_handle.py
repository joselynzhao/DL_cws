#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:pre_handle.py
@TIME:2019/8/19 14:43
@DES:
'''


import  re
import numpy as np

import codecs


def character_tagging(input_file, output_file):
    '''
    :param input_file: 原始语料文件
    :param output_file: 四词位标注的语料文件
    :return:
    '''
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "/s  ")
            else:
                output_data.write(word[0] + "/b  ")
                for w in word[1:len(word)-1]:
                    output_data.write(w + "/m  ")
                output_data.write(word[len(word)-1] + "/e  ")
            output_data.write("\n")
    input_data.close()
    output_data.close()



'''私有函数，不供外部调用'''
def clean(s):  # 整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])

'''私有函数，不供外部调用'''



# 收集字符集
'''
输入：用以收集字符的文件。。是【经过】tag处理的文件。输入文件类型是utf-8
输出：包含所有的字符集的文件。。输出文件类型也是用utf-8 

'''
def collect_chars(inputfile,chars_in_file,chars_out_file):
    '''
    :param inputfile: 词位标注后的语料文件
    :param chars_in_file: 字典文件 （可能非空）
    :param chars_out_file: 字典文件
    :return:
    '''
    chars_in = codecs.open(chars_in_file, 'r', 'utf-8')
    chars_out = codecs.open(chars_out_file,'w','utf-8')
    chars = chars_in.read().split(',')

    s = open(inputfile).read().decode('utf-8')
    # 输入的文件用utf-8来处理
    s = s.split('\r\n')
    s = u''.join(map(clean, s))
    s = re.split(u'[，。！？、]/[bems]', s)
    data = []  # 生成训练样本

    for i in s:
        x = get_xy(i)
        if x:
            data.append(x[0])

    for i in data:
        chars.extend(i)

    for char in chars:
        print char
        chars_out.write(char)

    # 关闭所有文件
    chars_in.close()
    chars_out.close()


if __name__ =="__main__":
    collect_chars("data_set/test_pku.txt","dictionary/char.txt","dictionary/char.txt")