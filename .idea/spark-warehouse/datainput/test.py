# #-*- coding:utf-8 -*-
# import tensorflow as tf
# import sys
# import numpy as np
#
# # tf.reduce_max()
# # with tf.Session() as sess:
# print(np.random.random())
# # from hdfs import *
# # client_N = Client("http://sjfx1:50070")
#
# from hdfs.client import Client #hdfs和本地文件的交互
# import pyhdfs as pd #判断文件是否存在
#
# fs_pyhdfs = pd.HdfsClient("sjfx1","50070")
# fs_hdfs = Client("http://"+"sjfx1"+":"+"50070")
# print(fs_pyhdfs.exists("/rezult/"+"AR"+"['G_ZDBY_0$', 'W', 'G_ZDBY_1_117NW001.1.txt|G_ZDBY_1_117NW002.1.txt|G_ZDBY_1_118NW001.1.txt|G_ZDBY_1_118NW002.1.txt|G_ZDBY_2_235NW001.1.txt|G_ZDBY_2_235NW002.1.txt|G_ZDBY_2_236NW001.1.txt|G_ZDBY_2_236NW002.1.txt', 593.0]"+".txt1"))
# sorted_list=[1,4,2,3]
# print(sorted(sorted_list))
#
# import tensorflow as tf
# temp_two=tf.constant([1,2])
# zero_out_module = tf.load_op_library('/tensorflow_user_lib/zero_out.so')
# with tf.Session(''):
#   print(zero_out_module.zero_out([[1, 2], [3, 4]]).eval())
#   print(temp_two.shape)
#

import re
import numpy as np

line = "2015-11-7 10:56:21.000000"
pattern = re.compile(r'(.*)\.([0-9]+)')
m = pattern.match(line)
if m:
  print(m.group(1))


import time
print(time.mktime(time.strptime(m.group(1),'%Y-%m-%d %H:%M:%S')))
print(time.mktime(time.strptime(m.group(1),'%Y-%m-%d %H:%M:%S')))
w=np.zeros([2,2])
w[1][1]=1
print(w)
