# 导入本地文件放入

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

import sample_model_sjfx
import AR_model_mapfunc
import KDE_model_mapfunc
import ekf_model_mapfunc

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

import pyhdfs as pd #判断文件是否存在
fs_pyhdfs = pd.HdfsClient("sjfx1","50070")

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
parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=5)
parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=9)
parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
parser.add_argument("-l", "--logdir", help="use log file on hdfs", default="abc")
args = parser.parse_args()

print("----------------AR  开始-------------------------------------")
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
FQW,cz_FQW=sample_model_sjfx.sample_from_hdfs(sc,hdfs_path=["/zd_data11.14/FQ/","/zd_data11.14/FS/","/zd_data11.14/FW/"],addrs="sjfx1",port="50070", \
                            group_num=2,sample_rato_FQS=1,sample_rato_FQS_cz=1,func=fuc)
sc.stop()

#准备inference用数据集
cz_FOW_inference=list.copy(cz_FQW)

def AR_model_start(sc,args,spark_worker_num,dataRDD,rdd_count,name):
    global client_N
    #['G_LYXGF', 'Q', 'G_LYXGF_1_315NQ001.S.txt|G_LYXGF_1_315NQ002.S.txt|G_LYXGF_1_316NQ001.S.txt|G_LYXGF_1_316NQ002.S.txt|G_LYXGF_1_317NQ001.S.txt|G_LYXGF_1_317NQ002.S.txt'
    num_executors = spark_worker_num
    num_ps = 0

    print("----------------AR-train start-------------------------------")
    #删除存储模型参数用目录
    if client_N.list("/user/root/").__contains__("model") and args.mode=='train':
       client_N.delete("/user/root/model/",recursive=True)

    print("args:",args)
    args.mode='train'
    print("{0} ===== Start".format(datetime.now().isoformat()))
    args.batch_size=int(rdd_count*0.90/spark_worker_num/2)

    cluster_AR_train = TFCluster.run(sc, AR_model_mapfunc.map_func_AR, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    # if args.mode == "train":
    cluster_AR_train.train(dataRDD, args.epochs)
    cluster_AR_train.shutdown()
    print("----------------AR-train over-------------------------------")

    #依次对每个站点的每个原地带入模型进行结果测算
    print("----------------AR-inference start--------------------------")
    #对所有测点进行一次遍历
    args.mode='inference'

    print("rdd count===============================",dataRDD.count())
    args.batch_size=int(rdd_count*0.90/spark_worker_num/2)
    print("args.batch_size=========================",args.batch_size)
    args.epochs=1
    print(args.mode)
    cluster_AR_inference = TFCluster.run(sc, AR_model_mapfunc.map_func_AR, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    labelRDD = cluster_AR_inference.inference(dataRDD).persist()
    labelRDD1 = labelRDD.filter(lambda x:not str(x[0]).__eq__('o')).persist()
    def func_m(partitionIndex,iter):
        num=0
        rezult=[]
        for i in iter:
            if num<10:
                rezult.append(["part:="+str(partitionIndex),i])
            num=num+1
        return rezult
    print("结果：==========================",labelRDD1.mapPartitionsWithIndex(func_m).collect())
    # print("labelRDD======luofeng:",labelRDD1.count())
    # .saveAsTextFile(args.output)
    cluster_AR_inference.shutdown()
    print("----------------AR-inference over--------------------------")

    # print("----------------KDE-train start------------------------")
    # print("labelRDD1 count===============================",labelRDD1.count())
    # args.mode='train'
    # cluster3 = TFCluster.run(sc,KDE_model_mapfunc.map_func_KDE, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    # args.epochs=1
    # cluster3.train(labelRDD1, args.epochs)
    # cluster3.shutdown()
    # print("----------------KDE-run over------------------------")

    print("----------------KDE-inference start------------------------")
    args.batch_size=int(labelRDD1.count()/spark_worker_num/2)
    args.mode='inference'
    print("args.batch_size=========================",args.batch_size)
    print("partition=========================",labelRDD1.getNumPartitions())
    cluster_KDE = TFCluster.run(sc,KDE_model_mapfunc.map_func_KDE, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    # if args.mode == "train":
    # labelRDD2=
    labelRDD3=cluster_KDE.inference(labelRDD1, args.epochs).persist()
    labelRDD4=labelRDD3.filter(lambda x:not str(x[0]).__eq__('o')).saveAsTextFile("hdfs://sjfx1:9000/rezult/"+str(name)+".txt")
    # print("labelRDD3:======",labelRDD4.take(100))
    # def func_m(partitionIndex,iter):
    #     num=0
    #     rezult=[]
    #     for i in iter:
    #         if num<100:
    #             rezult.append(["part:="+str(partitionIndex),i])
    #         num=num+1
    #     return rezult
    # print("结果：==========================",labelRDD3.mapPartitionsWithIndex(func_m).collect())
    cluster_KDE.shutdown()
    print("----------------KDE-inference over--------------------------")
    print("{0} ===== Stop".format(datetime.now().isoformat()))

#训练用抽取样本，测试用所有样本(正式使用版)
def AR_model_start_train(sc,args,spark_worker_num,dataRDD,rdd_count,name):
    global client_N
    #['G_LYXGF', 'Q', 'G_LYXGF_1_315NQ001.S.txt|G_LYXGF_1_315NQ002.S.txt|G_LYXGF_1_316NQ001.S.txt|G_LYXGF_1_316NQ002.S.txt|G_LYXGF_1_317NQ001.S.txt|G_LYXGF_1_317NQ002.S.txt'
    num_executors = spark_worker_num
    num_ps = 0

    print("----------------AR-train start-------------------------------")
    #删除存储模型参数用目录
    # if client_N.list("/user/root/").__contains__("model") and args.mode=='train':
    #     client_N.delete("/user/root/model/",recursive=True)
    #     client_N
    print("args:",args)
    args.mode='train'
    print("{0} ===== Start".format(datetime.now().isoformat()))
    args.batch_size=int((rdd_count-80)/spark_worker_num/10)
    print("args.batch_size:==============",args.batch_size)
    cluster_AR_train = TFCluster.run(sc, AR_model_mapfunc.map_func_AR, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    # if args.mode == "train":
    cluster_AR_train.train(dataRDD, args.epochs)
    cluster_AR_train.shutdown()
    print("----------------AR-train over-------------------------------")

def AR_model_start_inference(sc,args,spark_worker_num,dataRDD,name):

        global client_N
        #['G_LYXGF', 'Q', 'G_LYXGF_1_315NQ001.S.txt|G_LYXGF_1_315NQ002.S.txt|G_LYXGF_1_316NQ001.S.txt|G_LYXGF_1_316NQ002.S.txt|G_LYXGF_1_317NQ001.S.txt|G_LYXGF_1_317NQ002.S.txt'
        num_executors = spark_worker_num
        num_ps = 0

        #依次对每个站点的每个原地带入模型进行结果测算
        print("----------------AR-inference start--------------------------")
        #对所有测点进行一次遍历
        args.mode='inference'
        args.steps=1
        # print("rdd count===============================",dataRDD.count())
        def func_count(num,iter):
            j=0
            for i in iter:
                j=j+1
            return [j]
        each_length=dataRDD.mapPartitionsWithIndex(func_count).collect()
        print("每个partion的大小：===============", each_length)
        min_l=min(each_length)
        max_l=max(each_length)
        if min_l==0 and max_l==0:#不需要继续进行95%和5%分位点进行搜索
            pass
        else:
            each_length=sorted(list(each_length))
            for i_sort in range(each_length.__len__()):#查找最小minl非0值
                if each_length[i_sort] ==0:
                    pass
                else:
                    min_l=each_length[i_sort]
                    break
            if(max_l<min_l*1.5):
                if max_l>40000:
                    args.batch_size=int(numpy.average(each_length)/4)
                else:
                    args.batch_size=max_l
            else:
                if min_l>40000:
                    args.batch_size=min_l/2
                else:
                    args.batch_size=40000
        print("args.batch_size=========================",args.batch_size)
        args.epochs=1
        print(args.mode)
        cluster_AR_inference = TFCluster.run(sc, AR_model_mapfunc.map_func_AR, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
        labelRDD = cluster_AR_inference.inference(dataRDD)
        labelRDD1 = labelRDD.filter(lambda x:not str(x[0]).__eq__('o')).persist()
        def func_m(partitionIndex,iter):
            num=0
            rezult=[]
            for i in iter:
                if num<10:
                    rezult.append(["part:="+str(partitionIndex),i])
                num=num+1
            return rezult
        print("结果：==========================",labelRDD1.mapPartitionsWithIndex(func_m).collect())
        # labelRDD1.filter(lambda x:x[3]>500 or x[3]<-500).saveAsTextFile("hdfs://sjfx1:9000/rezult/"+"AR"+str(name)+"_500"+".txt")
        # labelRDD1=labelRDD1.filter(lambda x:x[3]<500 and x[3]>-500).persist()
        cluster_AR_inference.shutdown()
        print("----------------AR-inference over--------------------------")

        print("----------------KDE-inference start------------------------")
        def func_count(num,iter):
            j=0
            for i in iter:
                j=j+1
            return [j]
        each_length=labelRDD1.mapPartitionsWithIndex(func_count).collect()
        print("每个partion的大小：===============", each_length)
        min_l=min(each_length)
        max_l=max(each_length)
        #寻找数组不为0的最小点
        if min_l==0 and max_l==0:#不需要继续进行95%和5%分位点进行搜索
           pass
        else:
            each_length=sorted(list(each_length))
            for i_sort in range(each_length.__len__()):#查找最小minl非0值
                if each_length[i_sort] ==0:
                    pass
                else:
                    min_l=each_length[i_sort]
                    break
            if(max_l<min_l*1.5):
                if max_l>40000:
                    args.batch_size=int(numpy.average(each_length)/4)
                else:
                    args.batch_size=max_l
            else:
                if min_l>40000:
                    args.batch_size=min_l/2
                else:
                    args.batch_size=40000
                # exit()
            print("args.batch_size=========================",args.batch_size)
            args.epochs=1
            args.mode='inference'
            cluster_KDE = TFCluster.run(sc,KDE_model_mapfunc.map_func_KDE, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
            # if args.mode == "train":
            # labelRDD2=
            labelRDD3=cluster_KDE.inference(labelRDD1, args.epochs)
            labelRDD4=labelRDD3.filter(lambda x:not str(x[0]).__eq__('o')).saveAsTextFile("hdfs://sjfx1:9000/rezult/"+"AR_"+str(name)+".txt")
            # print("labelRDD3:======",labelRDD4.take(100))
            # def func_m(partitionIndex,iter):
            #     num=0
            #     rezult=[]
            #     for i in iter:
            #         if num<100:
            #             rezult.append(["part:="+str(partitionIndex),i])
            #         num=num+1
            #     return rezult
            # print("结果：==========================",labelRDD3.mapPartitionsWithIndex(func_m).collect())
            cluster_KDE.shutdown()
            print("----------------KDE-inference over--------------------------")
            print("{0} ===== Stop".format(datetime.now().isoformat()))


#启动进程，按每worker个为一组进行进行数据分解
num=0
spark_work=4
list_tmp=[]

#剔除厂站-原点大小小于500M的点，补足spark_work的点数量，用test名字代替（对齐运算用）
#print(client_N.status("/zd_data11.14/FQ/G_CFMY_1_001FQ001.txt"))
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


print("------------------")
cz_FQW=re+cz_FQW

# for i in cz_FQW:
#     print(i)

#print("需要处理的长度文件总长度=：",cz_FQW.__len__())
# 第一轮是进行模型训练，每个tensorflow custer训练一个模型
for i in list(cz_FQW):
    # if times==1:
    #     break
    if num==0:
        list_tmp.append(i)
        num=num+1
    else:
        if num%spark_work==0:
            bool=fs_pyhdfs.exists("/model/"+"AR_model_"+str(list_tmp[0][0])+"|"+str(list_tmp[0][1]))
            if(bool==False):
                sc=SparkContext(conf=conf)
                print(list_tmp)
                ex=sample_model_sjfx.sample_file_to_rdd(sc,filelist=list_tmp,work_num=spark_work,fractions=0.30,max_sample_length=10000,hdfs_addr="hdfs://sjfx1:9000/")
                rdd=sc.union(ex).persist()
                print("rdd.getNumPartitions:==============",rdd.getNumPartitions())
                rdd_count=rdd.count()
                print("rdd_count:=====================",rdd_count)
                AR_model_start_train(sc,args,spark_work,rdd,rdd_count,name=str(list_tmp[0][0]))
                sc.stop()
                print("-------------next AR_model_start--------------------")
                list_tmp=[]
                num=num+1
                list_tmp.append(i)
            else:
                print("-------------next AR_model_start--------------------")
                list_tmp=[]
                num=num+1
                list_tmp.append(i)
            # times=1
        else:
            list_tmp.append(i)
            num=num+1

print("last done：")#处理最后一组
print(list_tmp)
bool=fs_pyhdfs.exists("/model/"+"AR_model_"+str(list_tmp[0][0])+"|"+str(list_tmp[0][1]))
if(bool==False):
    sc=SparkContext(conf=conf)
    ex=sample_model_sjfx.sample_file_to_rdd(sc,filelist=list_tmp,fractions=0.30,max_sample_length=10000,work_num=spark_work,hdfs_addr="hdfs://sjfx1:9000/")
    rdd=sc.union(ex).persist()
    print("rdd.getNumPartitions:=",rdd.getNumPartitions())
    rdd_count=rdd.count()
    AR_model_start_train(sc,args,spark_work,rdd,rdd_count,name=list_tmp[0][0])
    sc.stop()
print("train all over")

# print("------------------AR  inference  start!-------------------------")
num=0
list_tmp=[]
print("------------------")

for i in list(cz_FQW):
    # if times==1:
    #     break
    if num==0:
        list_tmp.append(i)
        num=num+1
    else:
        if num%spark_work==0:
            bool=fs_pyhdfs.exists("/rezult/"+"AR_"+str(list_tmp[0][0])+"|"+str(list_tmp[0][1])+".txt")
            if bool==False:
                sc=SparkContext(conf=conf)
                ex=sample_model_sjfx.sample_file_to_rdd(sc,filelist=list_tmp,work_num=spark_work,fractions=0.30,max_sample_length=20000,hdfs_addr="hdfs://sjfx1:9000/")
                rdd=sc.union(ex).persist()
                AR_model_start_inference(sc,args,spark_work,rdd,name=str(list_tmp[0][0])+"|"+str(list_tmp[0][1]))
                sc.stop()
                print("-------------next AR_model_start--------------------")
                list_tmp=[]
                num=num+1
                list_tmp.append(i)
            else:
                print("-------------next AR_model_start--------------------")
                list_tmp=[]
                num=num+1
                list_tmp.append(i)
        else:
            list_tmp.append(i)
            num=num+1

print("last done：")#处理最后一组
bool=fs_pyhdfs.exists("/rezult/"+"AR_"+str(list_tmp[0][0])+"|"+str(list_tmp[0][1])+".txt")
if bool==False:
    sc=SparkContext(conf=conf)
    ex=sample_model_sjfx.sample_file_to_rdd(sc,filelist=list_tmp,work_num=spark_work,fractions=0.30,max_sample_length=20000,hdfs_addr="hdfs://sjfx1:9000/")
    rdd=sc.union(ex).persist()
    AR_model_start_inference(sc,args,spark_work,rdd,name=str(list_tmp[0][0])+"|"+str(list_tmp[0][1]))
    sc.stop()
print("AR_model all over")

#####efk模型 开始&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#二、数据样本抽样
#spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
print("----------------ekf_model 开始-------------------------------------")
args.steps=10
args.model="ekf"
sc=SparkContext(conf=conf)
def fuc_2(iterator):
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
FQW,cz_FQW=sample_model_sjfx.sample_from_hdfs(sc,hdfs_path=["/zd_data11.14/FQ/","/zd_data11.14/FS/","/zd_data11.14/FW/"],addrs="sjfx1",port="50070", \
                                         group_num=2,sample_rato_FQS=1,sample_rato_FQS_cz=1,func=fuc_2)
sc.stop()

#训练用抽取样本，测试用所有样本(正式使用版)
def ekf_model_start_train(sc,args,spark_worker_num,dataRDD,rdd_count,name):
    global client_N
    #['G_LYXGF', 'Q', 'G_LYXGF_1_315NQ001.S.txt|G_LYXGF_1_315NQ002.S.txt|G_LYXGF_1_316NQ001.S.txt|G_LYXGF_1_316NQ002.S.txt|G_LYXGF_1_317NQ001.S.txt|G_LYXGF_1_317NQ002.S.txt'
    num_executors = spark_worker_num
    num_ps = 0

    print("----------------ekf-train start-------------------------------")
    #删除存储模型参数用目录
    # if client_N.list("/user/root/").__contains__("model") and args.mode=='train':
    #     client_N.delete("/user/root/model/",recursive=True)
    #     client_N
    args.steps=5
    args.mode='train'
    print("{0} ===== Start".format(datetime.now().isoformat()))
    args.batch_size=int(rdd_count*0.90/spark_worker_num/20)
    if args.batch_size>500:
        args.batch_size=500
    else:
        pass

    cluster_AR_train = TFCluster.run(sc, ekf_model_mapfunc.map_func_ekf, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    # if args.mode == "train":
    cluster_AR_train.train(dataRDD, args.epochs)
    cluster_AR_train.shutdown()
    print("----------------ekf-train over-------------------------------")

def ekf_model_start_inference(sc,args,spark_worker_num,dataRDD,name):
    global client_N
    #['G_LYXGF', 'Q', 'G_LYXGF_1_315NQ001.S.txt|G_LYXGF_1_315NQ002.S.txt|G_LYXGF_1_316NQ001.S.txt|G_LYXGF_1_316NQ002.S.txt|G_LYXGF_1_317NQ001.S.txt|G_LYXGF_1_317NQ002.S.txt'
    num_executors = spark_worker_num
    num_ps = 0

    #依次对每个站点的每个原地带入模型进行结果测算
    print("----------------ekf-inference start--------------------------")
    #对所有测点进行一次遍历
    def func_count(num,iter):
        j=0
        for i in iter:
            j=j+1
        return [j]
    each_length=dataRDD.mapPartitionsWithIndex(func_count).collect()
    print("每个partion的大小：===============", each_length)
    min_l=min(each_length)
    max_l=max(each_length)
    if min_l==0 and max_l==0:#不需要继续进行95%和5%分位点进行搜索
        pass
    else:
        each_length=sorted(list(each_length))
        for i_sort in range(each_length.__len__()):#查找最小minl非0值
            if each_length[i_sort] ==0:
                pass
            else:
                min_l=each_length[i_sort]
                break
        if(max_l<min_l*1.5):
            if max_l>40000:
                args.batch_size=int(numpy.average(each_length)/4)
                    #int(numpy.average(each_length)/3)
            else:
                args.batch_size=max_l
        else:
            if min_l>40000:
                args.batch_size=min_l/2
            else:
                args.batch_size=40000
                    #20000
    print("args.batch_size=========================",args.batch_size)
    args.epochs=1
    args.mode='inference'
    args.model="ekf"
    cluster_AR_inference = TFCluster.run(sc, ekf_model_mapfunc.map_func_ekf, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    labelRDD = cluster_AR_inference.inference(dataRDD)
    labelRDD1 = labelRDD.filter(lambda x:not str(x[0]).__eq__('o')).persist()
    def func_m(partitionIndex,iter):
        num=0
        rezult=[]
        for i in iter:
            if num<10:
                rezult.append(["part:="+str(partitionIndex),i])
            num=num+1
        return rezult
    print("结果：==========================",labelRDD1.mapPartitionsWithIndex(func_m).collect())
    # def func_m(partitionIndex,iter):
    #     num=0
    #     rezult=[]
    #     for i in iter:
    #         if num<10:
    #             rezult.append(["part:="+str(partitionIndex),i])
    #         num=num+1
    #     return rezult
    # #剔除差异在-500 和 500以上的明显错误数据
    # labelRDD1.filter(lambda x:x[3]>500 or x[3]<-500).saveAsTextFile("hdfs://sjfx1:9000/rezult/"+"ekf"+str(name)+"_500"+".txt")
    # labelRDD1=labelRDD1.filter(lambda x:x[3]<500 and x[3]>-500).persist()
    #['G_CFYH|W', -1173.0488, 876.0, -2049.0488, 'G_CFYH_1_002FW001|2016-5-13 7:47:38.359000']
    cluster_AR_inference.shutdown()
    # print("结果：==========================",labelRDD1.mapPartitionsWithIndex(func_m).collect())
    # .saveAsTextFile(args.output)
    print("----------------ekf-inference over--------------------------")

    print("----------------KDE-inference start------------------------")
    def func_count(num,iter):
        j=0
        for i in iter:
            j=j+1
        return [j]
    each_length=labelRDD1.mapPartitionsWithIndex(func_count).collect()
    print("每个partion的大小：===============", each_length)
    min_l=min(each_length)
    max_l=max(each_length)
    if min_l==0 and max_l==0:#不需要继续进行95%和5%分位点进行搜索
        pass
    else:
        each_length=sorted(list(each_length))
        for i_sort in range(each_length.__len__()):#查找最小minl非0值
            if each_length[i_sort] ==0:
                pass
            else:
                min_l=each_length[i_sort]
                break
        if(max_l<min_l*1.5):
            if max_l>40000:
                args.batch_size=int(numpy.average(each_length)/4)
            else:
                args.batch_size=max_l
        else:
            if min_l>40000:
                args.batch_size=min_l/2
            else:
                args.batch_size=40000
    print("args.batch_size=========================",args.batch_size)
    args.epochs=1
    args.mode='inference'
    args.model='KDE'
    cluster_KDE = TFCluster.run(sc,KDE_model_mapfunc.map_func_KDE, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
    # if args.mode == "train":
    # labelRDD2=
    labelRDD3=cluster_KDE.inference(labelRDD1, args.epochs)
    labelRDD4=labelRDD3.filter(lambda x:not str(x[0]).__eq__('o')).saveAsTextFile("hdfs://sjfx1:9000/rezult/"+"ekf_"+str(name)+".txt")
    # # print("labelRDD3:======",labelRDD4.take(100))
    # def func_m(partitionIndex,iter):
    #     num=0
    #     rezult=[]
    #     for i in iter:
    #         if num<100:
    #             rezult.append(["part:="+str(partitionIndex),i])
    #         num=num+1
    #     return rezult
    # print("结果：==========================",labelRDD3.mapPartitionsWithIndex(func_m).collect())
    cluster_KDE.shutdown()
    print("----------------KDE-inference over--------------------------")
    print("{0} ===== Stop".format(datetime.now().isoformat()))

#启动进程，按每worker个为一组进行进行数据分解
num=0
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

#print("需要处理的长度文件总长度=：",cz_FQW.__len__())
# 第一轮是进行模型训练，每个tensorflow custer训练一个模型
for i in list(cz_FQW):
    # if times==1:
    #     break
    if num==0:
        list_tmp.append(i)
        num=num+1
    else:
        if num%spark_work==0:
            bool=fs_pyhdfs.exists("/model/"+"efk_model_"+str(list_tmp[0][0])+"|"+str(list_tmp[0][1]))
            if(bool==False):
                sc=SparkContext(conf=conf)
                ex=sample_model_sjfx.sample_file_to_rdd(sc,filelist=list_tmp,work_num=spark_work,fractions=0.30,max_sample_length=10000,hdfs_addr="hdfs://sjfx1:9000/")
                rdd=sc.union(ex).persist()
                rdd_count=rdd.count()
                ekf_model_start_train(sc,args,spark_work,rdd,rdd_count,name=list_tmp[0][0])
                sc.stop()
                print("-------------next efk_model_start--------------------")
                list_tmp=[]
                num=num+1
                list_tmp.append(i)
            else:
                print("-------------next efk_model_start--------------------")
                list_tmp=[]
                num=num+1
                list_tmp.append(i)
            # times=1
        else:
            list_tmp.append(i)
            num=num+1

print("efk_model last done：")#处理最后一组
bool=fs_pyhdfs.exists("/model/"+"efk_model_"+str(list_tmp[0][0])+"|"+str(list_tmp[0][1]))
if(bool==False):
    sc=SparkContext(conf=conf)
    ex=sample_model_sjfx.sample_file_to_rdd(sc,filelist=list_tmp,work_num=spark_work,fractions=0.30,max_sample_length=10000,hdfs_addr="hdfs://sjfx1:9000/")
    rdd=sc.union(ex).persist()
    ekf_model_start_train(sc,args,spark_work,rdd,rdd_count,name=list_tmp[0][0])
    sc.stop()
print("efk_model train all over")

# 用模型参数进行数据测试
num=0
list_tmp=[]
# print("aaa:==",cz_FQW)

for i in list(cz_FQW):
    # if times==1:
    #     break
    if num==0:
        list_tmp.append(i)
        num=num+1
    else:
        if num%spark_work==0:
            bool=fs_pyhdfs.exists("/rezult/"+"ekf_"+str(list_tmp[0][0])+"|"+str(list_tmp[0][1])+".txt")
            if bool==False:
                sc=SparkContext(conf=conf)
                ex=sample_model_sjfx.sample_file_to_rdd(sc,filelist=list_tmp,work_num=spark_work,fractions=0.50,max_sample_length=20000,hdfs_addr="hdfs://sjfx1:9000/")
                rdd=sc.union(ex).persist()
                ekf_model_start_inference(sc,args,spark_work,rdd,name=str(list_tmp[0][0])+"|"+str(list_tmp[0][1]))
                sc.stop()
                print("-------------next--------------------")
                list_tmp=[]
                num=num+1
                list_tmp.append(i)
                # times=1
            else:
                print("-------------next--------------------")
                list_tmp=[]
                num=num+1
                list_tmp.append(i)
        else:
            list_tmp.append(i)
            num=num+1

print("last done：")#处理最后一组
bool=fs_pyhdfs.exists("/rezult/"+"ekf_"+str(list_tmp[0][0])+"|"+str(list_tmp[0][1])+".txt")
if bool==False:
    sc=SparkContext(conf=conf)
    ex=sex=sample_model_sjfx.sample_file_to_rdd(sc,filelist=list_tmp,work_num=spark_work,fractions=0.50,max_sample_length=20000,hdfs_addr="hdfs://sjfx1:9000/")
    rdd=sc.union(ex).persist()
    ekf_model_start_inference(sc,args,spark_work,rdd,name=str(list_tmp[0][0])+"|"+str(list_tmp[0][1]))
    sc.stop()
print("ekf all over")


