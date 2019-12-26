# -*- coding: utf-8 -*-

#from segment import seg_one_img, load_dtc_module
from orderutils.recog_order import search_engine_recog, recog_order, recog_order_jieba
import time
import cv2
from PIL import Image
import numpy as np
import copy
import os
from itertools import permutations
from functools import reduce




# 求多个列表的组合
def combination(*lists): 
    total = reduce(lambda x, y: x * y, map(len, lists)) 
    retList = [] 
    for i in range(0, total): 
        step = total 
        tempItem = [] 
        for l in lists: 
            step /= len(l) 
            tempItem.append(l[int(i/step % len(l))]) 
        retList.append(tuple(tempItem)) 
    return retList 


# 使用新字典记录坐标,注意字典是无序的！！
def recordCoordinate(wordList, hanziList):
    center = {}
    for i in range(len(wordList)):
        center[wordList[i]] = [center for center in hanziList[i]][2]
    return center

# 破解函数


def get_order_list(all_hanzi_lists,hanzi_list,preds_list):
    #print(all_hanzi_lists)
    #print(hanzi_list)
    d = time.time()
    hanzi_combination = combination(*all_hanzi_lists)
    #print(hanzi_combination)
    hanzi_combination_connect = []
    for words in hanzi_combination:
        hanzi_combination_connect.append(''.join(words))

    # 识别语序
    hanzi_center = []
    jieba_flag = 0
    o = time.time()
    print('\n' * 2 + '语序识别' + '\n' + '*' * 80)
    for words in hanzi_combination_connect:  # 对每一个组合进行结巴分词
        # 此处对汉字的坐标进行记忆
        hanzi_center = recordCoordinate(words, hanzi_list)

        #print(words, 'jiaba')#mmmmmmmmmmmmm
        o = time.time()
        rec_word_possible = recog_order_jieba(words)
        if rec_word_possible:  # 如果遇到正确的词，则标志位置1
            jieba_flag = 1
            break
    if jieba_flag:
        rec_word = rec_word_possible
    else:
        #for words in hanzi_combination_connect:  
        ''.join(preds_list)
        print(preds_list)
        hanzi_center = recordCoordinate(preds_list, hanzi_list)
        #print(hanzi_center, 'engine')#mmmmmmmmmmmmmmm
        rec_word = search_engine_recog(preds_list)
        #rec_word = search_engine_recog(hanzi_combination_connect[0])
    print(preds_list)
    print('语序识别结果:{}'.format(rec_word))
    print('语序识别耗时{}'.format(time.time() - o))

    # 按正确语序输出坐标
    print('\n' * 2 + '最终结果' + '\n' + '*' * 80)
    centers = []
    #print(hanzi_center)
    for i in rec_word:
        centers.append(hanzi_center[i])
        
    
    print('正确语序的坐标：{}'.format(centers))
    
    print('总耗时{}'.format(time.time() - d))
    ##  调用时需要返回坐标
    return (centers),rec_word





if __name__ == '__main__':

    # 加载汉字定位模型
    all_hanzi_lists=[['世', '耳', '铺', '哺', '厘'],['大','太'], ['伞','平' , '部', '型', '播'], ['成','盛']]
    hanzi_list=[(b'hanzi1', 0.8764635920524597, (0.672152578830719, 0.355495423078537, 0.17341256141662598, 0.16976206004619598)), (b'hanzi2', 0.8573136329650879, (0.625790536403656, 0.7956624627113342, 0.15850003063678741, 0.13232673704624176)), (b'hanzi3', 0.857090175151825, (0.8480002284049988, 0.5595549941062927, 0.18965952098369598, 0.1373395025730133)), (b'hanzi4', 0.8561009168624878, (0.29499194025993347, 0.49679434299468994, 0.16142778098583221, 0.16253654658794403))]

    get_order_list(all_hanzi_lists, hanzi_list)




