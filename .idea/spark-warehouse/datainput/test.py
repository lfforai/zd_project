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

ww=[1,2,3,4,5,6]
print(ww[3:1])

# i="G_BJSZ_HH_01W_001FJ_PJ005.txt|G_BJSZ_HH_01W_002FJ_PJ005.txt|G_BJSZ_HH_01W_003FJ_PJ005.txt|G_BJSZ_HH_01W_004FJ_PJ005.txt|G_BJSZ_HH_01W_005FJ_PJ005.txt|G_BJSZ_HH_01W_006FJ_PJ005.txt|G_BJSZ_HH_01W_007FJ_PJ005.txt|G_BJSZ_HH_01W_008FJ_PJ005.txt|G_BJSZ_HH_01W_009FJ_PJ005.txt|G_BJSZ_HH_01W_010FJ_PJ005.txt|G_BJSZ_HH_01W_011FJ_PJ005.txt|G_BJSZ_HH_01W_012FJ_PJ005.txt|G_BJSZ_HH_01W_013FJ_PJ005.txt|G_BJSZ_HH_01W_014FJ_PJ005.txt|G_BJSZ_HH_01W_015FJ_PJ005.txt|G_BJSZ_HH_01W_016FJ_PJ005.txt|G_BJSZ_HH_01W_017FJ_PJ005.txt|G_BJSZ_HH_01W_018FJ_PJ005.txt|G_BJSZ_HH_01W_019FJ_PJ005.txt|G_BJSZ_HH_01W_020FJ_PJ005.txt|G_BJSZ_HH_01W_021FJ_PJ005.txt|G_BJSZ_HH_01W_022FJ_PJ005.txt|G_BJSZ_HH_01W_023FJ_PJ005.txt|G_BJSZ_HH_01W_024FJ_PJ005.txt|G_BJSZ_HH_01W_025FJ_PJ005.txt|G_BJSZ_HH_01W_026FJ_PJ005.txt|G_BJSZ_HH_01W_027FJ_PJ005.txt|G_BJSZ_HH_01W_028FJ_PJ005.txt|G_BJSZ_HH_01W_029FJ_PJ005.txt|G_BJSZ_HH_01W_030FJ_PJ005.txt|G_BJSZ_HH_01W_031FJ_PJ005.txt|G_BJSZ_HH_01W_032FJ_PJ005.txt|G_BJSZ_HH_01W_033FJ_PJ005.txt|G_BJSZ_HH_01W_034FJ_PJ005.txt|G_BJSZ_HH_01W_035FJ_PJ005.txt|G_BJSZ_HH_01W_036FJ_PJ005.txt|G_BJSZ_HH_01W_037FJ_PJ005.txt|G_BJSZ_HH_01W_038FJ_PJ005.txt|G_BJSZ_HH_01W_039FJ_PJ005.txt|G_BJSZ_HH_01W_040FJ_PJ005.txt|G_BJSZ_HH_01W_041FJ_PJ005.txt|G_BJSZ_HH_01W_042FJ_PJ005.txt|G_BJSZ_HH_01W_043FJ_PJ005.txt|G_BJSZ_HH_01W_044FJ_PJ005.txt|G_BJSZ_HH_01W_045FJ_PJ005.txt|G_BJSZ_HH_01W_046FJ_PJ005.txt|G_BJSZ_HH_01W_047FJ_PJ005.txt|G_BJSZ_HH_01W_048FJ_PJ005.txt|G_BJSZ_HH_01W_049FJ_PJ005.txt|G_BJSZ_HH_01W_050FJ_PJ005.txt|G_BJSZ_HH_01W_051FJ_PJ005.txt|G_BJSZ_HH_01W_052FJ_PJ005.txt|G_BJSZ_HH_01W_053FJ_PJ005.txt|G_BJSZ_HH_01W_054FJ_PJ005.txt|G_BJSZ_HH_01W_055FJ_PJ005.txt|G_BJSZ_HH_01W_056FJ_PJ005.txt|G_BJSZ_HH_01W_057FJ_PJ005.txt|G_BJSZ_HH_01W_058FJ_PJ005.txt|G_BJSZ_HH_01W_059FJ_PJ005.txt|G_BJSZ_HH_01W_060FJ_PJ005.txt|G_BJSZ_HH_01W_061FJ_PJ005.txt|G_BJSZ_HH_01W_062FJ_PJ005.txt|G_BJSZ_HH_01W_063FJ_PJ005.txt|G_BJSZ_HH_01W_064FJ_PJ005.txt|G_BJSZ_HH_01W_065FJ_PJ005.txt|G_BJSZ_HH_01W_066FJ_PJ005.txt|G_BJSZ_HH_01W_067FJ_PJ005.txt|G_BJSZ_HH_01W_068FJ_PJ005.txt|G_BJSZ_HH_01W_069FJ_PJ005.txt|G_BJSZ_HH_01W_070FJ_PJ005.txt|G_BJSZ_HH_01W_071FJ_PJ005.txt|G_BJSZ_HH_01W_072FJ_PJ005.txt|G_BJSZ_HH_01W_073FJ_PJ005.txt|G_BJSZ_HH_01W_074FJ_PJ005.txt|G_BJSZ_HH_01W_075FJ_PJ005.txt|G_BJSZ_HH_01W_076FJ_PJ005.txt|G_BJSZ_HH_01W_077FJ_PJ005.txt|G_BJSZ_HH_01W_078FJ_PJ005.txt|G_BJSZ_HH_01W_079FJ_PJ005.txt|G_BJSZ_HH_01W_080FJ_PJ005.txt|G_BJSZ_HH_01W_081FJ_PJ005.txt|G_BJSZ_HH_01W_082FJ_PJ005.txt|G_BJSZ_HH_01W_083FJ_PJ005.txt|G_BJSZ_HH_01W_084FJ_PJ005.txt|G_BJSZ_HH_01W_085FJ_PJ005.txt|G_BJSZ_HH_01W_086FJ_PJ005.txt|G_BJSZ_HH_01W_087FJ_PJ005.txt|G_BJSZ_HH_01W_088FJ_PJ005.txt|G_BJSZ_HH_01W_089FJ_PJ005.txt|G_BJSZ_HH_01W_090FJ_PJ005.txt|G_BJSZ_HH_01W_091FJ_PJ005.txt|G_BJSZ_HH_01W_092FJ_PJ005.txt|G_BJSZ_HH_01W_093FJ_PJ005.txt|G_BJSZ_HH_01W_094FJ_PJ005.txt|G_BJSZ_HH_01W_095FJ_PJ005.txt|G_BJSZ_HH_01W_096FJ_PJ005.txt|G_BJSZ_HH_01W_097FJ_PJ005.txt|G_BJSZ_HH_01W_098FJ_PJ005.txt|G_BJSZ_HH_01W_099FJ_PJ005.txt|G_BJSZ_HH_01W_100FJ_PJ005.txt"
# result=[]
#
# pitch=4
# list_value=str(i).split("|")#分解
# len=list_value.__len__()#长度
# print(len)
# yu=int(len%pitch)
# print(yu)
#
# if len<=pitch:#小于等于4个的情况下不进行分组
#   result.append(i)
# else:
#   pitch_num=int(len/pitch)
#   for j in range(pitch_num):
#     if j!=pitch_num-1:
#       temp=""
#       for w in range(pitch):
#         if w==0:
#           temp=str(temp)+str(list_value[j*pitch:j*pitch+pitch+1][w])+"|"
#         else:
#           if w!=pitch-1:
#             temp=str(temp)+str(list_value[j*pitch:j*pitch+pitch+1][w])+"|"
#           else:
#             temp=str(temp)+str(list_value[j*pitch:j*pitch+pitch+1][w])
#       result.append(["_$"+str(j),temp])
#     else:#余数
#       temp=""
#       for w in range(pitch+yu):
#         if w==0:
#           temp=str(temp)+str(list_value[j*pitch:j*pitch+pitch+1][w])+"|"
#         else:
#           if w!=pitch+yu-1:
#             temp=str(temp)+str(list_value[j*pitch:j*pitch+pitch+1][w])+"|"
#           else:
#             temp=str(temp)+str(list_value[j*pitch:j*pitch+pitch+1][w])
#       result.append(["_$"+str(j),temp])
#
# for i in result:
#   print(i)
#   print("---------------")
#
#   x=[1,2,4]
#   del x[2]
#   print(x)

i="G_GXJZ_GL_01W_001FJ_PW001.txt|G_GXJZ_GL_01W_002FJ_PW001.txt|G_GXJZ_GL_01W_003FJ_PW001.txt|G_GXJZ_GL_01W_004FJ_PW001.txt|G_GXJZ_GL_01W_005FJ_PW001.txt|G_GXJZ_GL_01W_006FJ_PW001.txt|G_GXJZ_GL_01W_007FJ_PW001.txt|G_GXJZ_GL_01W_008FJ_PW001.txt|G_GXJZ_GL_01W_009FJ_PW001.txt|G_GXJZ_GL_01W_010FJ_PW001.txt|G_GXJZ_GL_01W_011FJ_PW001.txt|G_GXJZ_GL_01W_012FJ_PW001.txt|G_GXJZ_GL_01W_013FJ_PW001.txt|G_GXJZ_GL_01W_014FJ_PW001.txt|G_GXJZ_GL_01W_015FJ_PW001.txt|G_GXJZ_GL_01W_016FJ_PW001.txt|G_GXJZ_GL_01W_017FJ_PW001.txt|G_GXJZ_GL_01W_018FJ_PW001.txt|G_GXJZ_GL_01W_019FJ_PW001.txt|G_GXJZ_GL_01W_020FJ_PW001.txt|G_GXJZ_GL_01W_021FJ_PW001.txt|G_GXJZ_GL_01W_022FJ_PW001.txt|G_GXJZ_GL_01W_023FJ_PW001.txt|G_GXJZ_GL_01W_024FJ_PW001.txt|G_GXJZ_GL_01W_025FJ_PW001.txt|G_GXJZ_GL_01W_026FJ_PW001.txt|G_GXJZ_GL_01W_027FJ_PW001.txt|G_GXJZ_GL_01W_028FJ_PW001.txt|G_GXJZ_GL_01W_029FJ_PW001.txt|G_GXJZ_GL_01W_030FJ_PW001.txt|G_GXJZ_GL_01W_031FJ_PW001.txt|G_GXJZ_GL_01W_032FJ_PW001.txt|G_GXJZ_GL_01W_033FJ_PW001.txt"
from hdfs.client import Client #hdfs和本地文件的交互
fs_hdfs = Client("http://"+"sjfx1"+":"+"50070")
list_value_N=list(str(i).split("|"))#分解
list_value=[e for e in list_value_N \
            if (fs_hdfs.status("/zd_data11.14/"+"PW"+"/"+str(e))['length'])/(1024)>2000]
print("aaa",list_value.__len__())
sss=[]
print(sss.__len__())