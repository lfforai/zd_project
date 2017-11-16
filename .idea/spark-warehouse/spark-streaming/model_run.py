# 导入本地文件放入
#/tool_lf/spark/spark-2.2.0-bin-hadoop2.7/sbin/spark-submit --py-files sample_model.py,AR_model_mapfunc.py,KDE_model_mapfunc.py  --conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" --conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4" model_run.py

# cd /
from pyspark.conf import SparkConf
import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

import sample_model
import AR_model_mapfunc
import KDE_model_mapfunc

from tensorflowonspark import TFCluster
import pyspark.sql as sql_n       #spark.sql
from pyspark import SparkContext  # pyspark.SparkContext dd
from pyspark.conf import SparkConf #conf

from hdfs import *
client_N = Client("http://127.0.0.1:50070")
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

#一、参数设置
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=10000)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
# parser.add_argument("-f", "--format", help="example format: (csv|pickle|tfr)", choices=["csv","pickle","tfr"], default="csv")
# parser.add_argument("-i", "--images", help="HDFS path to MNIST images in parallelized format")
# parser.add_argument("-l", "--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="AR_model")
parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=4)
parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=6)
parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=10)
parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
args = parser.parse_args()

#二、数据样本抽样
#spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
sc=SparkContext(conf=conf)
def fuc(iterator):
    value_list=[]
    for a in iterator:
        for j in range(str(a).__len__()):
            len=str(a).__len__()
            if j>0 and j<len-3:
                if  a[j].isdigit() and (a[j+1].__eq__("F") or a[j+1].__eq__("N")) \
                        and  (a[j+2].__eq__("W") or a[j+2].__eq__("Q") or a[j+2].__eq__("S")):
                    index2=str(a).find("_",2)
                    value_list.append([a[0:index2],a[j+2],a[0:j+1],str(a)])
    return value_list
FQW,cz_FQW=sample_model.sample_from_hdfs(sc,hdfs_path=["/zd_data11.14/FQ/","/zd_data11.14/FS/","/zd_data11.14/FW/"],addrs="127.0.0.1",port="50070", \
                            group_num=4,sample_rato_FQS=1,sample_rato_FQS_cz=1,func=fuc)
sc.stop()

def AR_model_start(sc,args,spark_worker_num,dataRDD,rdd_count):
    global client_N
    #['G_LYXGF', 'Q', 'G_LYXGF_1_315NQ001.S.txt|G_LYXGF_1_315NQ002.S.txt|G_LYXGF_1_316NQ001.S.txt|G_LYXGF_1_316NQ002.S.txt|G_LYXGF_1_317NQ001.S.txt|G_LYXGF_1_317NQ002.S.txt'
    num_executors = spark_worker_num
    num_ps = 0

    print("----------------AR-train start-------------------------------")
    #删除存储模型参数用目录
    if client_N.list("/user/root/").__contains__("model") and args.mode=='train':
        client_N.delete("/user/root/model/",recursive=True)

    print("args:",args)
    print("{0} ===== Start".format(datetime.now().isoformat()))
    args.batch_size=int(rdd_count*0.90/spark_worker_num/2)

    cluster = TFCluster.run(sc, AR_model_mapfunc.map_func, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    # if args.mode == "train":
    cluster.train(dataRDD, args.epochs)
    cluster.shutdown()
    print("----------------AR-train over-------------------------------")

    #依次对每个站点的每个原地带入模型进行结果测算
    print("----------------AR-inference start--------------------------")
    #对所有测点进行一次遍历
    args.mode='inference'
    args.batch_size=int(rdd_count*0.90/spark_worker_num/2)
    args.epochs=1
    print(args.mode)
    cluster1 = TFCluster.run(sc, AR_model_mapfunc.map_func, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    labelRDD = cluster1.inference(dataRDD)
    labelRDD1 = labelRDD.filter(lambda x:not str(x[0]).__eq__('o')).persist()
    def func_m(partitionIndex,iter):
        num=0
        rezult=[]
        for i in iter:
            if num<10:
                rezult.append(["part:="+str(partitionIndex),i])
            num=num+1
        return rezult
    print(labelRDD1.mapPartitionsWithIndex(func_m).collect())
    # print("labelRDD======luofeng:",labelRDD1.count())
    # .saveAsTextFile(args.output)
    cluster1.shutdown()
    print("----------------AR-inference over--------------------------")

    print("----------------KDE-inference start------------------------")
    cluster3 = TFCluster.run(sc,KDE_model_mapfunc.map_func, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    args.mode='train'
    cluster3.train(labelRDD1, args.epochs)
    cluster3.shutdown()

    args.batch_size=int(labelRDD1.count()*0.90/spark_worker_num/2)
    args.mode='inference'
    print("args.batch_size=========================",args.batch_size)
    cluster2 = TFCluster.run(sc,KDE_model_mapfunc.map_func, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    # if args.mode == "train":
    labelRDD2=cluster2.inference(labelRDD1, args.epochs)
    labelRDD2=labelRDD2.filter(lambda x:not str(x[0]).__eq__('o'))
    def func_m(partitionIndex,iter):
        num=0
        rezult=[]
        for i in iter:
            if num<100:
                rezult.append(["part:="+str(partitionIndex),i])
            num=num+1
        return rezult
    print(labelRDD2.mapPartitionsWithIndex(func_m).collect())
    cluster2.shutdown()
    print("----------------KDE-inference over--------------------------")
    print("{0} ===== Stop".format(datetime.now().isoformat()))

#启动进程，按每worker个为一组进行进行数据分解
num=0
spark_work=4
list_tmp=[]

#剔除厂站-原点大小小于500M的点，补足spark_work的点数量，用test名字代替（对齐运算用）
# print(client_N.status("/zd_data11.14/FQ/G_CFMY_1_001FQ001.txt"))
from operator import itemgetter, attrgetter
cz_FQW=list(filter(lambda x:x[3]>500,sorted(cz_FQW,key=itemgetter(3))))
yu_num=spark_work-cz_FQW.__len__()%spark_work
test_cz_FQW=cz_FQW[0]
test_cz_FQW[0]=cz_FQW[0][0]
new_cz_FQW=[test_cz_FQW]*yu_num
j=0
re=[]
for value in new_cz_FQW:
    re.append([value[0]+"_"+str(j)+"$",value[1],value[2],value[3]])
    j=j+1
cz_FQW=re+cz_FQW
# print(cz_FQW)

print("----------------开始-------------------------------------")
for i in list(cz_FQW):
    if num==0:
        list_tmp.append(i)
        num=num+1
    else:
        if num%spark_work==0:
            sc=SparkContext(conf=conf)
            ex=sample_model.sample_file_to_rdd(sc,filelist=list_tmp)
            rdd=sc.union(ex)
            print("rdd.getNumPartitions:=",rdd.getNumPartitions())
            rdd_count=rdd.count()
            AR_model_start(sc,args,4,rdd,rdd_count)
            print("******************************")
            # e[1].map(lambda x:[e[0],e[1]]).take(1)
            sc.stop()
            print("---------------------------------")
            list_tmp=[]
            num=num+1
            list_tmp.append(i)
        else:
            list_tmp.append(i)
            num=num+1
print("over")

