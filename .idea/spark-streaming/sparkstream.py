import tensorflow as tf
# with tf.device("/cpu:0"):
#     dataset = tf.contrib.data.Dataset.from_tensor_slices(
#         (tf.random_uniform([400000]),
#          tf.random_uniform([400000, 100], maxval=100, dtype=tf.int32)))
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = tf.Session(config=config)
#
#     def ccc(x,y):
#         return  tf.mod(y,2)
#     v=dataset.map(map_func=ccc,num_threads=500)
#     iterator = v.make_initializable_iterator()
#     next_element = iterator.get_next()
#
#     init = tf.global_variables_initializer()
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#
#     with tf.Session(config=config) as sess:
#         sess.run(iterator.initializer)
#         sess.run(init)
#         print(sess.run(next_element))

# 导入本地文件放入
from pyspark.conf import SparkConf
import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

from tensorflowonspark import TFCluster
import pyspark.sql as sql_n       #spark.sql
from pyspark import SparkContext  # pyspark.SparkContext dd
from pyspark.conf import SparkConf #conf

from pyspark.sql.types import *
schema = StructType([
    StructField("id",  StringType(), True),
    StructField("value", FloatType(), True),
    StructField("date", StringType(), True)]
)

os.environ['JAVA_HOME'] = "/tool_lf/java/jdk1.8.0_144/bin/java"
os.environ["PYSPARK_PYTHON"] = "/root/anaconda3/bin/python"
os.environ["HADOOP_USER_NAME"] = "root"
conf=SparkConf().setMaster("spark://lf-MS-7976:7077")
# os.environ['JAVA_HOME'] = conf.get(SECTION, 'JAVA_HOME')
spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
sc =spark.sparkContext
sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

# 路径
import multiprocessing
import time
import pyhdfs as pd
import numpy as np
fs = pd.HdfsClient("127.0.0.1", 9000)

#n阶移动平均
n_e_s=5
times_e_s=1 #2,3
first_e_s=5
a=0.5
def exponential_smoothing(iterator):
   #一次指数平滑
   if times_e_s==1:
        global n_e_s
        rezult_list=[]
        num=0
        pre_value=0
        value_first=0
        value_list=[]
        pre=0
        for i in iterator:
            if num<first_e_s:
                value_first=float(i)+value_first
                value_list.append(float(i))
                num=num+1
            else:
                if num==first_e_s:
                   pre_value=value_first/first_e_s
                   for j in range(first_e_s):
                        rezult_list.append([value_list[j],pre_value])
                        pre_value=a*value_list[j]+(1-a)*pre_value
                else:
                   rezult_list.append([float(i),a*float(i)+(1-a)*pre_value])
                   pre_value=a*float(i)+(1-a)*pre_value
        return rezult_list

#n阶指数平滑法
n_olymic=5
def olymic(iterator):
    global n_olymic
    rezult_list=[]
    value_list=[]
    num=0
    for i in iterator:
        if num%n_olymic!=0 or num==0:
            value_list.append(float(i))
            num=num+1
        else:
            rezult_list.append([float(i),np.round(np.average(value_list),decimals=2)])
            value_list=[]
            num=0
    return rezult_list

#n阶段加权平滑法
n_moving=5
weight=[0.3,0.25,0.2,0.15,0.1]
def moving(iterator):
    global n_moving
    rezult_list=[]
    value_list=[]
    weight_n=np.array(weight)
    num=0
    for i in iterator:
        if num%n_moving!=0 or num==0:
            value_list.append(float(i))
            num=num+1
        else:
            rezult_list.append([float(i),np.sum(np.array(value_list)*weight_n,axis=-1)])
            value_list=[]
            num=0
    return rezult_list

#开始剔除异常数据
# print(sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001.txt") \
#       .map(lambda x:str(x).split(",")).filter(lambda x:float(x[1])>0).map(lambda x:float(x[1])) \
#       .count())

print(sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001.txt")\
    .map(lambda x:str(x).split(",")).filter(lambda x:float(x[1])>0).map(lambda x:float(x[1]))\
    .mapPartitions(olymic).filter(lambda x:x[0]/x[1]>2).take(100))

print(sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001.txt") \
      .map(lambda x:str(x).split(",")).filter(lambda x:float(x[1])>0).map(lambda x:float(x[1])) \
      .mapPartitions(exponential_smoothing).filter(lambda x:x[0]/x[1]>2).take(100))

print(sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001.txt") \
      .map(lambda x:str(x).split(",")).filter(lambda x:float(x[1])>0).map(lambda x:float(x[1])) \
      .mapPartitions(moving).filter(lambda x:x[0]/x[1]>2).collect())
# print(np.sum(np.array([1,2,3])*np.array([1,2,3])))
