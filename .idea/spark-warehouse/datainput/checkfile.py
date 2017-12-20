
import os

# def file_name(file_dir):
#     rezult=[]
#     for root, dirs, files in os.walk(file_dir):
#         rezult.append(files)
#     return rezult
# a=file_name('/lf/data/test1')
# b=file_name('/lf/data/test')
#
# file=open('/lf/data/数据检查相关/title1','w')
# for i in range(list(a[0]).__len__()):
#    file.write(str(a[0][i]+"\n"));
#
# file=open('/lf/data/数据检查相关/title2','w')
# for i in range(list(b[0]).__len__()):
#     file.write(str(b[0][i]+"\n"));
# file.close()

from pyspark.conf import SparkConf
import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time


from datetime import datetime
from hdfs.client import Client #hdfs和本地文件的交互
import pyhdfs as pd #判断文件是否存在
import numpy as np

addrs="sjfx1"
port="50070"

fs_pyhdfs = pd.HdfsClient(addrs,port)
fs_hdfs = Client("http://"+addrs+":"+port)

list_file=[]
for e in fs_hdfs.list("/zd_data11.14/PJ/"):
    if fs_hdfs.status("/zd_data11.14/"+"PJ"+"/"+str(e))['length']<100:
        list_file.append(e)
for e in fs_hdfs.list("/zd_data11.14/PW/"):
    if fs_hdfs.status("/zd_data11.14/"+"PW"+"/"+str(e))['length']<100:
        list_file.append(e)
for e in fs_hdfs.list("/zd_data11.14/CU/"):
    if fs_hdfs.status("/zd_data11.14/"+"CU"+"/"+str(e))['length']<100:
        list_file.append(e)

for e in fs_hdfs.list("/zd_data11.14/CU/"):
    if fs_hdfs.status("/zd_data11.14/"+"CU"+"/"+str(e))['length']<100:
        list_file.append(e)

file=open('/lf/data/数据检查相关/数据较少点名单2','w')
for i in range(list_file.__len__()):
    file.write(str(list_file[i]+"\n"));
#
print(list_file)


#对电量指标进行测试判断是否有异常点
from pyspark.conf import SparkConf
import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

# import sample_model_sjfx
# import AR_model_mapfunc
# import KDE_model_mapfunc
# import ekf_model_mapfunc
# import Clustering_model_mapfunc
# import spearman_ttf_model_mapfunc

from tensorflowonspark import TFCluster
import pyspark.sql as sql_n       #spark.sql
from pyspark import SparkContext  # pyspark.SparkContext dd
from pyspark.conf import SparkConf #conf

from hdfs import *
client_N = Client("http://sjfx1:50070")
from pyspark.sql.types import *

schema = StructType([
    StructField("id",  StringType(), True),
    StructField("value", FloatType(), True),
    StructField("date", StringType(), True)]
)

os.environ['JAVA_HOME'] = "/tool_lf/java/jdk1.8.0_144/bin/java"
os.environ["PYSPARK_PYTHON"] = "/root/anaconda3/bin/python"
os.environ["HADOOP_USER_NAME"] = "root"
conf=SparkConf().setMaster("spark://sjfx4:7077")

import os
result=[]
fd =open("/lf/data/数据检查相关/spear", "r" )
for line in fd.readlines():
    result.append(line.replace("\n",""))

def find_PW(filename):
    len=str(filename).__len__()
    index=0
    for i in range(5):
        index=str(filename).find("_",index+1)
    type_N=str(filename)[index+1:index+3]
    if(str(type_N).__eq__("PW")):
      return True
    else:
      return False

import pyhdfs as pd #判断文件是否存在
fs_pyhdfs = pd.HdfsClient("sjfx1","50070")
# a=fs_pyhdfs.exists("/zd_data11.14/PJ/G_DBBT_SY_01W_009FJ_PJ002.txt")
# print(a)
# exit()
sc=SparkContext(conf=conf)
n=0
for e in result:
    if find_PW(e):
       # print(fs_pyhdfs.exists("/zd_data11.14/PJ/"+str(e).replace("PW","PJ")+".txt"))
       print("--------------------------")
       print("/zd_data11.14/PJ/"+str(e)+".txt")
       if(fs_pyhdfs.exists("/zd_data11.14/PJ/"+str(e).replace("PW","PJ")+".txt")):

           rdd_PJ=sc.textFile("hdfs://sjfx1:9000/zd_data11.14/PJ/"+str(e).replace("PW","PJ")+".txt")
           list_PJ=rdd_PJ.take(20)
           if(len(list_PJ)==20):
               PJ=float(str(list_PJ[19]).split(",")[1])-float(str(list_PJ[0]).split(",")[1])
               time_start=str(list_PJ[0]).split(",")[2]
               time_end=str(list_PJ[19]).split(",")[2]

               time_start=time.mktime(time.strptime(time_start,'%Y-%m-%d %H:%M:%S'))
               time_end=time.mktime(time.strptime(time_end,'%Y-%m-%d %H:%M:%S'))

               rdd_PW=sc.textFile("hdfs://sjfx1:9000/zd_data11.14/PW/"+str(e)+".txt")
               rdd_PW=rdd_PW.filter(lambda x:time_start<=time.mktime(time.strptime(str(x).split(",")[2],'%Y-%m-%d %H:%M:%S'))
                                      and time.mktime(time.strptime(str(x).split(",")[2],'%Y-%m-%d %H:%M:%S'))<=time_end)\
                                      .map(lambda x:float(str(x).split(",")[1])).collect()
               if(len(rdd_PW)>0):
                 PJ_N=sum(list(rdd_PW))*(time_end-time_start)/list(rdd_PW).__len__()/3600
                 print("rezult:={%f},{%f}"%(PJ,PJ_N))
               else:
                 print("PW数据太少，无法验证")
           else:
               print("PJ数据太少，无法验证")
       n=n+1
print(n)


