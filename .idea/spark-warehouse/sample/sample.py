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

# from pyspark.context import SparkContext
# sc_tf = SparkContext(conf=conf)

# 路径
import multiprocessing
import time
import pyhdfs as pd
import numpy as np
fs = pd.HdfsClient("127.0.0.1", 9000)

#把字符串拆解为
#['G_NMWL_2_080FQ001', 'G_NMWL', 'G_NMWL_2_080|FQ001']
def fuc(iterator):
    value_list=[]
    for a in iterator:
        for j in range(str(a).__len__()):
            len=str(a).__len__()
            if j>0 and j<len-3:
                if  a[j].isdigit() and (a[j+1].__eq__("F") or a[j+1].__eq__("N")) \
                        and  (a[j+2].__eq__("W") or a[j+2].__eq__("Q") or a[j+2].__eq__("S")):
                        index2=str(a).find("_",2)
                        value_list.append([a[0:j+1],a[0:index2]+"|"+a[0:j+1]+"|"+str(a)])
    return value_list

rdd=sc.textFile("hdfs://127.0.0.1:9000/test_dir/title.txt")\
    .map(lambda x:str(x).replace("\'",""))\
    .map(lambda x:str(x).split(",")[0])\
    .mapPartitions(fuc)

def add(x,y):
   return str(x)+"|"+str(y)

#['G_NMWL_2_080FQ001', 'G_NMWL', 'G_NMWL_2_080|FQ001']
def fuc2(iterator):
    value_list=[]
    for a in iterator:
        index2=str(a[0]).find("_",2)
        value_list.append([str(a[0])[0:index2],a[1]])
    return value_list

rdd=rdd.map(lambda x:[x[0],(str(x[1]).split("|"))[2]]).sortByKey(lambda x:x[0])\
    .reduceByKey(add).mapPartitions(fuc2)
print(rdd.count())
fractions=dict(rdd.map(lambda x:x[0]).distinct().map(lambda x:(x,0.03)).collect())
rdd=rdd.sampleByKey(withReplacement=False,fractions=fractions,seed=0)
rdd.flatMap(lambda x:str(x[1]).split("|")).repartition(1).saveAsTextFile("hdfs://127.0.0.1:9000/test_dir/sample.txt")


#
# print(rdd.map(lambda x:[x[0],(str(x[1]).split("|"))[2]]).reduceByKey(add)
#       .take(1000))
    # sampleByKey(withReplacement = false,0.1,seed=None)


