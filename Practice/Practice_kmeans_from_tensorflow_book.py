# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:47:49 2017

@author: 14224
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

""" Define parameters """
#群集所有的點數為1000
num_vectors = 1000
#分區的數量
num_clusters = 4
#演算法的計算步階
num_steps = 100

""" Data """
#initialize the data
x_values = []
y_values = []
vector_values = []

#create random data
#numpy 的 random.normal允許我們建立x_values和y_values向量
for i in xrange(num_vectors):
    if np.random.random() > 0.5:
        x_values.append(np.random.normal(0.4,0.7))
        y_values.append(np.random.normal(0.2,0.8))
    else:
        x_values.append(np.random.normal(0.6,0.4))
        y_values.append(np.random.normal(0.8,0.5))

#使用zip函數來獲得完整串列
vector_values = zip(x_values, y_values)

#透過tensorflow將vector_values轉換為一個常數
vectors = tf.constant(vector_values)

#視覺化
plt.plot(x_values, y_values, 'o', label = 'Input Data')
plt.legend()
plt.show()