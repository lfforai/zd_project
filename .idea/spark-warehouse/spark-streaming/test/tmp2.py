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

def map_fnc(iter):
    rezult=[]
    num=0
    for i in iter:
        num=num+1
    rezult.append(num)
    return rezult

rdd=sc.parallelize(range(100000000))
print(rdd.getNumPartitions())
print(rdd.mapPartitions(map_fnc).collect())
print(sc.parallelize(range(100)).repartition(1).collect())
      # .mapPartitions(map_fnc).repartition(1).collect())


# y=np.array([16,36,46,56,56,56,65])
# pi=3.1415926
# x=54
# h=1
# print(np.mean(np.exp(-np.power(x-y,2)/(np.power(h,2)*2.0))/(np.power(pi*2.0,0.5)*h)))
# #tf.reduce_sum(tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.linspace(1,1,n))))
# print(np.linspace(1,3,3))