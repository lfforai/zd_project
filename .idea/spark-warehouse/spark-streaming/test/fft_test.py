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
import re



import pyspark.sql as sql_n       #spark.sql
from pyspark import SparkContext  # pyspark.SparkContext dd
from pyspark.conf import SparkConf #conf
os.environ['JAVA_HOME'] = "/tool_lf/java/jdk1.8.0_144/bin/java"
os.environ["PYSPARK_PYTHON"] = "/root/anaconda3/bin/python"
os.environ["HADOOP_USER_NAME"] = "root"
conf=SparkConf().setMaster("spark://sjfx4:7077")
# os.environ['JAVA_HOME'] = conf.get(SECTION, 'JAVA_HOME')
spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
sc =spark.sparkContext
sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

from math import sqrt

def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

def corrcoef(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den

#在一个ｒｄｄ中提取一段连续长短的数据
def rdd_catch_pitch(sc,filename,length=100000,start_point=-1,time_point="#"):
    import numpy as np

    rdd=sc.textFile(filename)

    count_num=rdd.count()#总长度
    if length>count_num:
        print("采样的长度不能大于样本全部长度！")
        exit()

    def func_count(num,iter):
        j=0
        for i in iter:
            j=j+1
        return [j]
    each_length=list(rdd.mapPartitionsWithIndex(func_count).collect())
    # print(each_length)

    #每个partion开始编号的地方
    each_length_N=[]
    total_num=0;
    for i in range(each_length.__len__()):
        if i==0:
           each_length_N.append(0)
           total_num=each_length[i]
        else:
           each_length_N.append(total_num)
           total_num=total_num+each_length[i]

    #开始编码ｉｎｄｅｘ
    def map_index(each_length_N):
      def map_func(num,iter):
        start=each_length_N[num]
        n=0
        rezult=[]
        for i in iter:
            rezult.append([n+start,i])
            n=n+1
        return rezult
      return map_func
    rdd=rdd.mapPartitionsWithIndex(map_index(each_length_N)).persist()
    # print(rdd.take(10))

    if  start_point!=-1:
        a=rdd.filter(lambda x:x[0]==start_point).collect()
        time_point=str(a[0][1]).split(",")[2]
    else:
        pattern = re.compile(r'(.*)\.([0-9]+)')
        m = pattern.match(time_point)
        time_n=time.mktime(time.strptime(m.group(1),'%Y-%m-%d %H:%M:%S'))
        def time_func(time_n):
          def map_func(iter):
            rezult=[]
            pattern = re.compile(r'(.*)\.([0-9]+)')
            for i in iter:
                value=i[1].split(",")
                m=pattern.match(str(value[2]))
                liunx_time=time.mktime(time.strptime(m.group(1),'%Y-%m-%d %H:%M:%S'))
                if liunx_time-time_n>60:
                   break
                if liunx_time-time_n<60 and liunx_time-time_n>=0:
                   rezult.append(i)
                   break
            return rezult
          return map_func
        a=rdd.mapPartitions(time_func(time_n)).collect()
        start_point=int(a[0][0])
        print(start_point)

    # 开始采样点
    rdd=rdd.filter(lambda x:x[0]>=start_point and x[0]<start_point+length).\
        map(lambda x:[x[0],float(str(x[1]).split(",")[1])])

    rezult=rdd.collect()

    #####################
    result=[]
    for i in [[float(e[1].real),float(e[1].imag)] for e in rezult]:
        if np.abs(i[0])<0.5:
            result.append(0)
        else:
            result.append(i[0])

        if np.abs(i[1])<0.5:
            result.append(0)
        else:
            result.append(i[1])

    sc.stop
    return result,time_point

rdd0,time_N=numpy.array(rdd_catch_pitch(sc,"hdfs://sjfx1:9000/zd_data11.14/FW/G_CFMY_1_001FW001.txt",start_point=-1,time_point="2015-11-7 10:56:21.000000"))
rdd1,_=numpy.array(rdd_catch_pitch(sc,"hdfs://sjfx1:9000/zd_data11.14/FW/G_CFMY_1_002FW001.txt",start_point=-1,time_point=time_N))
rdd2,_=numpy.array(rdd_catch_pitch(sc,"hdfs://sjfx1:9000/zd_data11.14/FW/G_CFMY_1_003FW001.txt",start_point=-1,time_point=time_N))
rdd3,_=numpy.array(rdd_catch_pitch(sc,"hdfs://sjfx1:9000/zd_data11.14/FW/G_CFMY_1_001FW001_QQ.txt",start_point=-1,time_point=time_N))
rdd4,_=numpy.array(rdd_catch_pitch(sc,"hdfs://sjfx1:9000/zd_data11.14/FW/G_CFMY_1_063FW001.txt",start_point=-1,time_point=time_N))
rdd5,_=numpy.array(rdd_catch_pitch(sc,"hdfs://sjfx1:9000/zd_data11.14/FW/G_CFMY_1_064FW001.txt",start_point=-1,ttime_point=time_N))
#rdd6=numpy.asarray(rdd_catch_pitch(sc,"hdfs://sjfx1:9000/zd_data11.14/FS/G_CFYH_2_062FS001.txt",1000))
temp_shape=tf.zeros([1])#传递shape的参数

with tf.Session() as sess:
    pearson_out_module = tf.load_op_library('/tensorflow_user_lib/pearson_out.so')
    p_value_up1=sess.run(pearson_out_module.pearson_out(rdd0,rdd3,sess.run(temp_shape)))#传递shape的参数))
    p_value_up2=sess.run(pearson_out_module.pearson_out(rdd1,rdd3,sess.run(temp_shape)))#传递shape的参数))
    p_value_up3=sess.run(pearson_out_module.pearson_out(rdd2,rdd3,sess.run(temp_shape)))#传递shape的参数))
    p_value_up4=sess.run(pearson_out_module.pearson_out(rdd1,rdd4,sess.run(temp_shape)))#传递shape的参数))
    p_value_up5=sess.run(pearson_out_module.pearson_out(rdd2,rdd5,sess.run(temp_shape)))#传递shape的参数))
    p_value_up6=sess.run(pearson_out_module.pearson_out(rdd1,rdd2,sess.run(temp_shape)))#传递shape的参数))
    p_value_up7=sess.run(pearson_out_module.pearson_out(rdd4,rdd5,sess.run(temp_shape)))#传递shape的参数))

print(p_value_up1)
print(p_value_up2)
print(p_value_up3)
print(p_value_up4)
print(p_value_up5)
print(p_value_up6)
print(p_value_up7)
print("-----------------------------")
# print(corrcoef(rdd0,rdd3))
# print(corrcoef(rdd0,rdd5))
# print(corrcoef(rdd1,rdd3))
# print(corrcoef(rdd1,rdd5))
# print(corrcoef(rdd2,rdd3))
# print(corrcoef(rdd2,rdd5))
# print("-----------------")
# print(corrcoef(rdd0,rdd1))
# print(corrcoef(rdd1,rdd2))
