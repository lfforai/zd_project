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
# spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
# sc =spark.sparkContext
# sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)


#一、#############################################################################
#文件名分解函数，把原点名分解为原点名字、厂站、原点设备
#['G_CFMY', 'Q', 'G_CFMY_1_001', 'G_CFMY_1_001FQ001.txt']
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

#采样可能来源于不同的文件夹FQ，FS，FW，hdfs_path是一个总体文件的目录
def sample_from_hdfs(sc,hdfs_path=["/zd_data11.14/FQ/","/zd_data11.14/FS/","/zd_data11.14/FW/"],addrs="127.0.0.1",port="50070",\
                     group_num=4,sample_rato_FQS=1.0,sample_rato_FQS_cz=1.0,func= lambda x:x):

      #当func对文件名字进行分组的规则
         #分组规则1：按每个FQ，FS，FW每个文件夹中的各厂站数据文件进行抽样,建立全部总体模型
         #分组规则2：按每个厂为一组，使用所有文件，但是对每个文件中的数据进行抽样建模

      from hdfs.client import Client #hdfs和本地文件的交互
      import pyhdfs as pd #判断文件是否存在
      import numpy as  np

      fs_pyhdfs = pd.HdfsClient(addrs,port)
      fs_hdfs = Client("http://"+addrs+":"+port)

      #全部样本 ['G_CFMY', 'Q', 'G_CFMY_1_001', 'G_CFMY_1_001FQ001.txt']
      total_fname_list=[]
      for i in hdfs_path:
          total_fname_list.extend([e for e in func(fs_hdfs.list(i))])

      rdd=sc.parallelize(total_fname_list).map(lambda x:[str(x[0])+"|"+str(x[1]),x[3]])

      #按FQ，FS，FW分层抽样：每个厂站都必须有原点样本，其中抽样比率按sample_rato
      fractions=dict(rdd.map(lambda x:x[0]).distinct().map(lambda x:(x,sample_rato_FQS)).collect())

      list_total=rdd.sampleByKey(withReplacement=False,fractions=fractions,seed=0).collect()

      #FQ，FW，FS
      FQ_total=map(lambda x:x[1],filter(lambda x:str(x[0]).split("|")[1].__eq__("Q"),list_total))
      FS_total=map(lambda x:x[1],filter(lambda x:str(x[0]).split("|")[1].__eq__("S"),list_total))
      FW_total=map(lambda x:x[1],filter(lambda x:str(x[0]).split("|")[1].__eq__("W"),list_total))

      group_name_total_list=[FQ_total,FS_total,FW_total]

      #按每个厂站抽样
      def add(x,y):
        return str(x)+"|"+str(y)

      fractions=dict(rdd.map(lambda x:x[0]).distinct().map(lambda x:(x,sample_rato_FQS_cz)).collect())
      group_name_cz_list=rdd.sampleByKey(withReplacement=False,fractions=fractions,seed=0).reduceByKey(add)\
      .map(lambda x:[str(x[0]).split("|")[0],str(x[0]).split("|")[1],x[1]]).collect()
      return  group_name_total_list,group_name_cz_list
       #group_name_total_list:['G_LYXGF', 'W', 'G_LYXGF_1_115NW001.1.txt|G_LYXGF_1_115NW002.1.txt|G_LYXGF_1_116NW001.1.txt|G_LYXGF_1_116NW002.1.txt|G_LYXGF_1_117NW001.1.txt|G_LYXGF_1_117NW002.1.txt']
       #group_name_cz_list: ['G_LYXGF', 'W', 'G_LYXGF_1_115NW001.1.txt|G_LYXGF_1_115NW002.1.txt|G_LYXGF_1_116NW001.1.txt|G_LYXGF_1_116NW002.1.txt|G_LYXGF_1_117NW001.1.txt|G_LYXGF_1_117NW002.1.txt']

#厂站-QSW
def sample_file_to_rdd(sc,filedir="/zd_data11.14/",filelist="",work_num=4,fractions=0.30,max_sample_length=50000):

    def rdd_sample(fractions,ep_len,max_length):
        import numpy as np
        import random
        #在每个rdd的每个partition中按fractions的比例进行样本抽样
        #抽样比例：fractions,每个partiton的预估比例ep_len
        #时序模型需要，抽取连续的区间
        def map_func(iter):
            start_point=ep_len*(1-fractions)*np.random.random()#开始取样点
            length=fractions*ep_len#抽样长度
            if length>max_length:#最大抽样长度
                 length=max_length
            result=[]
            num=0
            for i in iter:
                if num>start_point and num<length:
                    value=str(i).split(",")
                    result.append([str(value[0])+"|"+str(value[2]),float(value[1])])
                num=num+1
            return result
        return map_func

    yd_num=list(filelist).__len__()
    all_rdd_list=[]#所有点的list集合
    if(yd_num==work_num):#需要拟合的点数量正好等于work数量
        for i in filelist:
            cz_name=i[0]#厂站名字
            eq_type=i[1]#原点种类 F功率 Q电量 S风速
            filename_list=[filedir+"F"+str(eq_type)+"/"+str(e) for e in str(i[2]).split("|")]
            # print(filename_list)
            cz_rdd_list=[]#每个厂+Q，W，F
            for j in filename_list:
              #每个原点按照比例进行抽样
              rdd_tmp=sc.textFile(j)
              partitions_num=rdd_tmp.getNumPartitions()
              total_count=rdd_tmp.count()
              each_max_limit=max_sample_length/partitions_num
              eachpartitions_len=int(total_count/partitions_num*0.9)
              rdd_tmp=rdd_tmp.mapPartitions(rdd_sample(fractions,eachpartitions_len,each_max_limit)).map(lambda x:[cz_name+"|"+eq_type,x])#进行抽样,partition的顺序会被打乱,但是每个partition内部顺序不动
              cz_rdd_list.append(rdd_tmp)
            all_rdd_list.append(sc.union(cz_rdd_list).repartition(1))
    else:
       print("一次输入的厂站-QFW数量必须和spark的worker数量一致")
    return  all_rdd_list

#二、############################################################################
from dateutil import parser
#处理将不规范的日期调整成规范日期
spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
sc =spark.sparkContext
rdd=sc.textFile("hdfs://127.0.0.1:9000/zd_data11.14/FQ/G_CFMY_1_001FQ001.txt").map(lambda x:str(x).split(",")) \
    .map(lambda x:[x[0],x[1],str(parser.parse(str(x[2])))])

#日期查看日期是否有错误,2015-12-09 22:22:23 980 2015-12-09 22:22:25 1980 2015-12-09 22:22:26 980 2015-12-09 22:22:27 1980
def fuc(iterator):
    value_list=[]
    num=0
    pre=0
    list_five_value=[]
    rezult=[]
    for i in iterator:
        if num==0:
           list_five_value.append(i)
        else:
            if num%5==0 and num !=0:
               a=float(list_five_value[1][0])-float(list_five_value[0][0])
               b=float(list_five_value[2][0])-float(list_five_value[1][0])
               c=float(list_five_value[3][0])-float(list_five_value[2][0])
               d=float(list_five_value[4][0])-float(list_five_value[3][0])
               if(a*b<0 and b*c<0 and d*c<0):
                   rezult.append(list_five_value)
                   list_five_value=[]
                   list_five_value.append(i)
               else:
                   list_five_value=[]
                   list_five_value.append(i)
            else:
               list_five_value.append(i)
        num=num+1
    return rezult
spark.stop()
#print(rdd.sortBy(lambda x:x[2]).map(lambda x:[x[1],x[2]]).mapPartitions(fuc).take(100))
#错误数据：[[['6802171.0', '2016-01-16 19:02:13'], ['0.0', '2016-01-16 19:02:20'], ['6802171.0', '2016-01-16 19:02:26'], ['0.0', '2016-01-16 19:02:33'], ['6802171.0', '2016-01-16 19:02:39']]]

