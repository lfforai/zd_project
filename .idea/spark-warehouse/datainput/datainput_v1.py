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
conf=SparkConf().setMaster("spark://sjfx4:7077")
# os.environ['JAVA_HOME'] = conf.get(SECTION, 'JAVA_HOME')
spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
sc =spark.sparkContext
sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

#将单个文件上传hadoop集群
def local_to_hdfs(hdfs_path="/zd_data11.14/",addrs="127.0.0.1",port="50070",local_filename="",file_hdfsname=""):
    import multiprocessing
    import time

    from hdfs.client import Client #hdfs和本地文件的交互
    import pyhdfs as pd #判断文件是否存在

    fs_pyhdfs = pd.HdfsClient(addrs,port)
    fs_hdfs = Client("http://"+addrs+":"+port)

    if(file_hdfsname.__contains__("CU") or file_hdfsname.__contains__("CU")):
        mid_path="CU/"
    else:
        if(file_hdfsname.__contains__("PJ") or file_hdfsname.__contains__("PJ")):
            mid_path="PJ/"
        else:
            mid_path="PW/"

    if(not fs_pyhdfs.exists(hdfs_path+mid_path+file_hdfsname)):
        print("存储开始："+hdfs_path+mid_path+file_hdfsname)
        print("local path:=",local_filename)
        fs_pyhdfs.copy_from_local(local_filename,hdfs_path+mid_path+file_hdfsname)
        print(hdfs_path+mid_path+file_hdfsname)
    else:
        print(hdfs_path+mid_path+file_hdfsname+"exsit!")

#将整个文件夹中文件上传
def local_dir_to_hdfs(hdfs_path="/zd_data11.14/",addrs="127.0.0.1",port="50070",local_filedir=""):
    #遍历文件夹
    def bl_file(file_dir=local_filedir):
        [files]=os.walk(file_dir)
        files_list=list(files)
        list_fl={"FS":"FS","FQ":"FQ","FW":"FW"}
        files_local=[item for item in map(lambda x:str(file_dir)+str(x),files_list[2])]
        files_name=[item for item in files_list[2]]
        print("file_loacl:=",files_local)
        print("file_name:=",files_name)
        return files_local,files_name  #返回本地文件名字（路径）和不带路进

    # 数据导入
    local,hdfs =bl_file(file_dir=local_filedir)
    # pool = multiprocessing.Pool(processes = 3)
    result=[]
    print("本次共导入文件：",len(local))
    for i  in range(len(local)):
        local_to_hdfs(hdfs_path,addrs,port,local[i],hdfs[i])
    print("Sub-process(es) done.")

#删除文件
def delete_hdfs(hdfs_path="/zd_data11.14/",addrs="127.0.0.1",port="50070",recursive=True):
    from hdfs.client import Client #hdfs和本地文件的交互
    import pyhdfs as pd #判断文件是否存在

    fs_pyhdfs = pd.HdfsClient(addrs,port)
    fs_hdfs = Client("http://"+addrs+":"+port)
    if fs_pyhdfs.exists(hdfs_path):
       fs_pyhdfs.delete(hdfs_path,recursive=recursive)
       print("hdfs://"+addrs+":"+port+"/zd_data11.14/"+"is deleted!")
    else:
       print("hdfs://"+addrs+":"+port+"/zd_data11.14/"+"is not exsit!")

# def local_all_to_hdfs:

#×××××××××××××××××××××××××××××文件目录××××××××××××××××××××××××××××××
#文件或者目录删除
deletefilename="/rezult"
    #"/rezult/AR['G_ZDBY', 'Q', 'G_ZDBY_1_117NQ001.D.txt|G_ZDBY_1_117NQ002.D.txt|G_ZDBY_1_118NQ001.D.txt|G_ZDBY_1_118NQ002.D.txt|G_ZDBY_2_235NQ001.D.txt|G_ZDBY_2_235NQ002.D.txt|G_ZDBY_2_236NQ001.D.txt|G_ZDBY_2_236NQ002.D.txt', 604.0].txt"
    #"/user/root/model"
    #"/rezult"

#目录上传
local_to_hdfs_dirnames="AR['G_CFYH', 'Q', 'G_CFYH_1_002FQ001.txt|G_CFYH_1_003FQ001.txt|G_CFYH_1_004FQ001.txt|G_CFYH_1_005FQ001.txt|G_CFYH_2_060FQ001.txt|G_CFYH_2_061FQ001.txt|G_CFYH_2_062FQ001.txt', 8544.0]_500.txt" #文件夹上传

#单个文件上传
local_dir="/lf/2017.11.14/test_data/"
local_to_hdfs_filename="G_CFMY_1_001FQ001_S.txt"

hdfs_dir="/zd_data11.14/FQ/"
file_hdfsname="G_CFMY_1_001FQ001_S.txt"

#整个文件夹上传
local_dir="/lf/data/test1/"

hdfs_dir="/zd_data11.14/"

################################执行命令###############################
#删除文件
if True:
   delete_hdfs(hdfs_path=deletefilename,addrs="sjfx1",port="50070")

if False:
#单个本地文件上传
   local_to_hdfs(hdfs_path=hdfs_dir,addrs="sjfx1",port="50070",local_filename=local_dir+local_to_hdfs_filename,file_hdfsname=file_hdfsname)

if False:
#整个本地文件上传
   local_dir_to_hdfs(hdfs_path=hdfs_dir,addrs="sjfx1",port="50070",local_filedir=local_dir)

# sc.textFile("hdfs://sjfx1:9000/zd_data11.14/FQ/G_CFYH_1_004FQ001_W.txt")\
#     .map(lambda x:str(x).replace("G_CFYH_1_004FW001","G_CFYH_1_004FQ001_W"))\
#     .saveAsTextFile("hdfs://sjfx1:9000/zd_data11.14/FQ/G_CFYH_1_004FQ001_WW.txt")
# sc.textFile("hdfs://sjfx1:9000/G_CFYH_1_004FQ001_W.txt")
#

# sc.textFile("hdfs://sjfx1:9000/zd_data11.14/FQ/G_CFMY_1_001FQ001_S.txt") \
#     .map(lambda x:str(x).replace("G_CFMY_1_001FS001","G_CFMY_1_001FQ001_S")) \
#     .saveAsTextFile("hdfs://sjfx1:9000/zd_data11.14/FQ/G_CFMY_1_001FQ001_SS.txt")

# sc.textFile("hdfs://sjfx1:9000/zd_data11.14/FS/G_CFMY_1_001FS001_W.txt") \
#     .map(lambda x:str(x).replace("G_CFMY_1_001FW001","G_CFMY_1_001FS001_W")) \
#     .saveAsTextFile("hdfs://sjfx1:9000/zd_data11.14/FS/G_CFMY_1_001FS001_WW.txt")
#

# sc.textFile("hdfs://sjfx1:9000/zd_data11.14/FS/G_CFYH_1_005FS001_Q.txt") \
#     .map(lambda x:str(x).replace("G_CFYH_1_005FQ001","G_CFYH_1_005FS001_Q")) \
#     .saveAsTextFile("hdfs://sjfx1:9000/zd_data11.14/FS/G_CFYH_1_005FS001_QQ.txt")

# sc.textFile("hdfs://sjfx1:9000/zd_data11.14/FW/G_CFMY_1_001FW001_Q.txt") \
#     .map(lambda x:str(x).replace("G_CFMY_1_001FQ001","G_CFMY_1_001FW001_Q")) \
#     .saveAsTextFile("hdfs://sjfx1:9000/zd_data11.14/FW/G_CFMY_1_001FW001_QQ.txt")

# sc.textFile("hdfs://sjfx1:9000/zd_data11.14/FW/G_CFYH_1_004FW001_S.txt") \
#     .map(lambda x:str(x).replace("G_CFYH_1_004FS001","G_CFYH_1_004FW001_S")) \
#     .saveAsTextFile("hdfs://sjfx1:9000/zd_data11.14/FW/G_CFYH_1_004FW001_SS.txt")