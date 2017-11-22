#-*- coding:utf-8 -*-
import tensorflow as tf
import sys
import numpy as np

# tf.reduce_max()
# with tf.Session() as sess:
print(np.random.random())
# from hdfs import *
# client_N = Client("http://sjfx1:50070")

from hdfs.client import Client #hdfs和本地文件的交互
import pyhdfs as pd #判断文件是否存在

fs_pyhdfs = pd.HdfsClient("sjfx1","50070")
fs_hdfs = Client("http://"+"sjfx1"+":"+"50070")
print(fs_pyhdfs.exists("/rezult/"+"AR"+"['G_ZDBY_0$', 'W', 'G_ZDBY_1_117NW001.1.txt|G_ZDBY_1_117NW002.1.txt|G_ZDBY_1_118NW001.1.txt|G_ZDBY_1_118NW002.1.txt|G_ZDBY_2_235NW001.1.txt|G_ZDBY_2_235NW002.1.txt|G_ZDBY_2_236NW001.1.txt|G_ZDBY_2_236NW002.1.txt', 593.0]"+".txt1"))