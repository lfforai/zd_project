#///////////////////////////////////////////////////////////////////////////////////////

#./spark-submit --conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" --conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4"  ~/IdeaProjects/pyspark_t/.idea/spark-warehouse/spark-streaming/exp.py



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
#
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from hdfs import *
client_N = Client("http://127.0.0.1:50070")

import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

from tensorflowonspark import TFCluster
import AR_model_use
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader
# import pyspark.sql as sql_n       #spark.sql
# from pyspark import SparkContext  # pyspark.SparkContext dd
# from pyspark.conf import SparkConf #conf

os.environ['JAVA_HOME'] = "/tool_lf/java/jdk1.8.0_144/bin/java"
os.environ["PYSPARK_PYTHON"] = "/root/anaconda3/bin/python"
os.environ["HADOOP_USER_NAME"] = "root"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["spark.executorEnv.LD_LIBRARY_PATH"]="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"
os.environ["spark.executorEnv.CLASSPATH"]="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}"
os.environ["spark.executorEnv.HADOOP_HDFS_HOME"]="/tool_lf/hadoop/hadoop-2.7.4"

conf=SparkConf().setMaster("spark://titianx:7077")
sc=SparkContext(conf=conf)
# spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
# sc =spark.sparkContext
# sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

executors = sc._conf.get("spark.executor.instances")
print("executors:=",executors)
num_executors = int(executors) if executors is not None else 4
num_ps = 0

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=10000)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
# parser.add_argument("-f", "--format", help="example format: (csv|pickle|tfr)", choices=["csv","pickle","tfr"], default="csv")
# parser.add_argument("-i", "--images", help="HDFS path to MNIST images in parallelized format")
# parser.add_argument("-l", "--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="AR_model")
parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=4)
parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=1)
parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
args = parser.parse_args()

#删除存储模型参数用目录
if client_N.list("/user/root/").__contains__("model") and args.mode=='train':
    client_N.delete("/user/root/model/",recursive=True)

print("args:",args)
print("{0} ===== Start".format(datetime.now().isoformat()))

def sample_map(fraction_base,rato):
  def _sample_map(iter):
    while True:
      fraction_use=random.random()
      if fraction_use-rato<0.10 and fraction_use-rato>-0.10:
          break
    input_length=int(fraction_base*fraction_use)
    rezult=[]
    num=0
    # start=random.random()*1000
    for i in iter:
      if num<input_length and 240<num:
        rezult.append(i)
      else:
        if num>input_length:
         break
      num=num+1
    return rezult
  return _sample_map


# dataRDD1_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(1,float(x[1]))).count()
# dataRDD2_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_035FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(2,float(x[1]))).count()
# dataRDD3_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(3,float(x[1]))).count()
# dataRDD4_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_041FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(4,float(x[1]))).count()
# dataRDD5_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(5,float(x[1]))).count()
# dataRDD6_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_035FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(6,float(x[1]))).count()
# dataRDD7_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(7,float(x[1]))).count()
# dataRDD8_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_041FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(8,float(x[1]))).count()

fraction_base,rato=100000,0.75
dataRDD1=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt")\
.map(lambda x:str(x).split(",")).map(lambda x:(1,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
dataRDD2=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_035FS001.txt") \
    .map(lambda x:str(x).split(",")).map(lambda x:(2,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
dataRDD3=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt") \
    .map(lambda x:str(x).split(",")).map(lambda x:(3,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
dataRDD4=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_041FS001.txt") \
    .map(lambda x:str(x).split(",")).map(lambda x:(4,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
# dataRDD5=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(5,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
# dataRDD6=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_035FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(6,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
# dataRDD7=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(7,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
# dataRDD8=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_041FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(8,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)

a=[dataRDD1,dataRDD2,dataRDD3,dataRDD4]
    #,dataRDD5,dataRDD6,dataRDD7,dataRDD8]
dataRDD=sc.union(a)
print(dataRDD.take(100))
rdd_count=dataRDD.count()
print("count:====================",rdd_count)
print("partition:=",dataRDD.getNumPartitions())
# def func(x,iter):
#     result = []
#     for value in iter:
#         result.append((x,value))
#     return result
# dataRDD=sc.parallelize(range(150003),3).mapPartitionsWithIndex(func)
# print(dataRDD.take(100))

if rdd_count<100000:
   args.epochs=2
   args.batch_size=rdd_count
else:
   args.epochs=1
   args.batch_size=100000

# print("getNumPartitions:=",dataRDD.getNumPartitions())
cluster = TFCluster.run(sc, map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
# if args.mode == "train":
cluster.train(dataRDD, args.epochs)
cluster.shutdown()
print("-----------------train over-------------------------------")
# # else:
def func1(iter):
    result = []
    num=0
    for value in iter:
        result.append(value)
        num=num+1
        if num>240:
            break
    return result
dataRDD1=sc.union(a).mapPartitions(func1)
# dataRDD1=sc.parallelize(range(600),3).mapPartitionsWithIndex(func)
# print("getNumPartitions:=",dataRDD1.getNumPartitions())
args.mode='inference'
args.batch_size=300
args.epochs=1
print(args.mode)
cluster1 = TFCluster.run(sc, map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
labelRDD = cluster1.inference(dataRDD1)
print(labelRDD.filter(lambda x:not str(x[0]).__eq__('o')).collect())# .saveAsTextFile(args.output)
cluster1.shutdown()
print("-----------------inference over-------------------------------")
print("{0} ===== Stop".format(datetime.now().isoformat()))