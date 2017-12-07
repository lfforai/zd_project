# 导入本地文件放入
#./spark-submit --py-files ~/IdeaProjects/zd_project/.idea/spark-warehouse/spark-streaming/sample_model.py  --py-files  ~/IdeaProjects/zd_project/.idea/spark-warehouse/spark-streaming/AR_model_mapfunc.py  --conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" --conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4"  ~/IdeaProjects/zd_project/.idea/spark-warehouse/spark-streaming/model_run.py

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
def sample_from_hdfs(sc,hdfs_path=["/zd_data11.14/FQ/","/zd_data11.14/FS/","/zd_data11.14/FW/"],addrs="sjfx1",port="50070",\
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
      group_name_cz_list=[[e[0],e[1],e[2],\
                        np.round(np.sum([fs_hdfs.status("/zd_data11.14/"+"F"+str(e[1])+"/"+str(value))['length']/np.power(1024,2) for value  in str(e[2]).split("|")]),0)] for e in group_name_cz_list]
      return  group_name_total_list,group_name_cz_list
       #group_name_total_list:['G_LYXGF', 'W', 'G_LYXGF_1_115NW001.1.txt|G_LYXGF_1_115NW002.1.txt|G_LYXGF_1_116NW001.1.txt|G_LYXGF_1_116NW002.1.txt|G_LYXGF_1_117NW001.1.txt|G_LYXGF_1_117NW002.1.txt']
       #group_name_cz_list: ['G_LYXGF', 'W', 'G_LYXGF_1_115NW001.1.txt|G_LYXGF_1_115NW002.1.txt|G_LYXGF_1_116NW001.1.txt|G_LYXGF_1_116NW002.1.txt|G_LYXGF_1_117NW001.1.txt|G_LYXGF_1_117NW002.1.txt']

#厂站-QSW
def sample_file_to_rdd(sc,filedir="/zd_data11.14/",filelist=[],work_num=4,fractions=0.50,max_sample_length=10000,hdfs_addr="hdfs://sjfx1:9000"):

    def rdd_sample(fractions,ep_len,max_length,is_Q=True):
        import numpy as np
        import random
        #在每个rdd的每个partition中按fractions的比例进行样本抽样
        #抽样比例：fractions,每个partiton的预估比例ep_len
        #时序模型需要，抽取连续的区间
        def map_func(iter):
            if is_Q!=True:#不是电量
                length=min(int(ep_len*fractions),max_length)
                start_point=int((ep_len-length-1)*np.random.random())#剩余长度中随机选者一个点作为开始点
                result=[]
                num=0
                for i in iter:
                    if num>start_point and num<start_point+length:
                        value=str(i).split(",")
                        result.append([str(value[0])+"|"+str(value[2]),float(value[1])])
                    num=num+1
            else:#是电量求增量
                length=min(int(ep_len*fractions),max_length)
                start_point=int((ep_len-length-1)*np.random.random())
                result=[]
                num=0
                pre=0.0
                for i in iter:
                    if num>start_point and num<start_point+length:
                        if num ==start_point+1:
                           value=str(i).split(",")
                           pre=float(value[1])
                        else:
                           value=str(i).split(",")
                           temp=float(value[1])-pre
                           if(temp!=0):
                             result.append([str(value[0])+"|"+str(value[2]),temp])
                             pre=float(value[1])
                           else:
                             pre=float(value[1])
                             pass
                    num=num+1
            return result
        return map_func

    yd_num=list(filelist).__len__()
    print("本次处理点个数：=",yd_num)
    all_rdd_list=[]#所有点的list集合
    if(yd_num==work_num):#需要拟合的点数量正好等于work数量
        for i in filelist:
           cz_name=i[0]#厂站名字
           eq_type=i[1]#原点种类 F功率 Q电量 S风速
           file_length=float(i[3])
           filename_list=[hdfs_addr+filedir+"F"+str(eq_type)+"/"+str(e) for e in str(i[2]).split("|")]
           # print(filename_list)
           cz_rdd_list=[]#每个厂+Q，W，F
           sum_count=0
           for j in filename_list:
              #每个原点按照比例进行抽样
              rdd_tmp=sc.textFile(j)
              partitions_num=rdd_tmp.getNumPartitions()
              total_count=rdd_tmp.count()
              # sum_count=sum_count+total_count
              each_max_limit=max_sample_length/partitions_num
              eachpartitions_len=int(total_count/partitions_num*0.9)
              if  eq_type=="Q":
                  rdd_tmp=rdd_tmp.mapPartitions(rdd_sample(fractions,eachpartitions_len,each_max_limit*5,True)).map(lambda x:[cz_name+"|"+eq_type,x])#进行抽样,partition的顺序会被打乱,但是每个partition内部顺序不动
                  # print("电量增量:范例：==",rdd_tmp.take(100))
              else:
                  rdd_tmp=rdd_tmp.mapPartitions(rdd_sample(fractions,eachpartitions_len,each_max_limit,False)).map(lambda x:[cz_name+"|"+eq_type,x])#进行抽样,partition的顺序会被打乱,但是每个partition内部顺序不动
              cz_rdd_list.append(rdd_tmp)
           all_rdd_list.append(sc.union(cz_rdd_list).repartition(1))
    else:
       print("一次输入的厂站-QFW数量必须和spark的worker数量一致")
    return  all_rdd_list

#用来做聚类使用
#默认１００个点为一个组
def cluster_file_to_rdd(sc,filedir="/zd_data11.14/",filelist=[],work_num=4,fractions=0.50,max_sample_length=100,hdfs_addr="hdfs://sjfx1:9000",pitch_length=500):

    def rdd_sample(ep_len,max_length,pitch_len):#max_lengt每个partition抽取的个数
        import numpy as np
        import random
        from numpy import std
        #预先剔除2倍标准差以外的数据
        #在每个rdd的每个partition中按fractions的比例进行样本抽样
        #抽样比例：fractions,每个partiton的预估比例ep_len
        #时序模型需要，抽取连续的区间
        def map_func(iter):
            length=pitch_len*max_length#总抽样长度
            start_point=int((ep_len-length-1)*np.random.random())
            result=[]
            value=[]
            num=0
            j=0#pitch循环用
            for i in iter:
                if num>start_point and num<start_point+length+1:
                    if j%pitch_len==0 and j!=0:
                       one=np.array(value[1:value.__len__()])
                       std_value=std(one)
                       avg_value=np.average(one)
                       result.append([value[0],avg_value,std_value])
                       value=[]
                       a=i.split(",")
                       value.append(str(a[0]))
                       value.append(float(a[1]))
                       j=1
                    else:
                       if j==0:
                          value=[]
                          a=i.split(",")
                          value.append(str(a[0]))
                          value.append(float(a[1]))
                          j=j+1
                       else:
                          a=i.split(",")
                          value.append(float(a[1]))
                          j=j+1
                num=num+1
            return result
        return map_func

    yd_num=list(filelist).__len__()
    print("本次处理点个数：=",yd_num)
    all_rdd_list=[]#所有点的list集合
    if(yd_num==work_num):#需要拟合的点数量正好等于work数量
        for i in filelist:
            cz_name=i[0]#厂站名字
            eq_type=i[1]#原点种类 F功率 Q电量 S风速
            file_length=float(i[3])
            filename_list=[hdfs_addr+filedir+"F"+str(eq_type)+"/"+str(e) for e in str(i[2]).split("|")]
            # print(filename_list)
            cz_rdd_list=[]#每个厂+Q，W，F
            sum_count=0
            for j in filename_list:
                #每个原点按照比例进行抽样
                rdd_tmp=sc.textFile(j)
                partitions_num=rdd_tmp.getNumPartitions()
                total_count=rdd_tmp.count()
                # sum_count=sum_count+total_count
                each_max_limit=max_sample_length/partitions_num
                eachpartitions_len=int(total_count/partitions_num*0.9)
                if  each_max_limit*pitch_length>eachpartitions_len:
                    print("每次抽样的样本总数需求不能大于ｒｄｄ的partition长度")
                    exit()
                rdd_tmp=rdd_tmp.mapPartitions(rdd_sample(eachpartitions_len,each_max_limit,pitch_length))#进行抽样,partition的顺序会被打乱,但是每个partition内部顺序不动
                # print("list':=",rdd_tmp.take(1))
                cz_rdd_list.append(rdd_tmp)
            all_rdd_list.append(sc.union(cz_rdd_list).repartition(1))
    else:
        print("一次输入的厂站-QFW数量必须和spark的worker数量一致")
        exit()
    return  all_rdd_list

#快速傅利叶变换提取主成分
def cluster_FFT_file_to_rdd(sc,filedir="/zd_data11.14/",filelist=[],work_num=4,fractions=0.50,max_sample_length=100,hdfs_addr="hdfs://sjfx1:9000",pitch_length=100):

    def rdd_sample(ep_len,max_length,pitch_len):#max_lengt每个partition抽取的个数
        import numpy as np
        import random
        from numpy import std
        #预先剔除2倍标准差以外的数据
        #在每个rdd的每个partition中按fractions的比例进行样本抽样
        #抽样比例：fractions,每个partiton的预估比例ep_len
        #时序模型需要，抽取连续的区间
        def map_func(iter):
            length=pitch_len*max_length#总抽样长度
            start_point=int((ep_len-length-1)*np.random.random())
            result=[]
            value=[]
            num=0
            j=0#pitch循环用
            for i in iter:
                if num>start_point and num<start_point+length+1:
                    if j%pitch_len==0 and j!=0:
                        one=np.array(value[1:value.__len__()])
                        vlaue_fft=np.fft.fft(one)
                        fft_value=np.sum(vlaue_fft[0:3]*[0.6,0.3,0.1])##取前面３个主要比重
                        result.append([value[0],fft_value.real,fft_value.imag])
                        value=[]
                        a=i.split(",")
                        value.append(str(a[0]))
                        value.append(float(a[1]))
                        j=0
                    else:
                        if j==0:
                            value=[]
                            a=i.split(",")
                            value.append(str(a[0]))
                            value.append(float(a[1]))
                            j=j+1
                        else:
                            a=i.split(",")
                            value.append(float(a[1]))
                            j=j+1
                num=num+1
            return result
        return map_func

    yd_num=list(filelist).__len__()
    print("本次处理点个数：=",yd_num)
    all_rdd_list=[]#所有点的list集合
    if(yd_num==work_num):#需要拟合的点数量正好等于work数量
        for i in filelist:
            cz_name=i[0]#厂站名字
            eq_type=i[1]#原点种类 F功率 Q电量 S风速
            file_length=float(i[3])
            filename_list=[hdfs_addr+filedir+"F"+str(eq_type)+"/"+str(e) for e in str(i[2]).split("|")]
            # print(filename_list)
            cz_rdd_list=[]#每个厂+Q，W，F
            sum_count=0
            for j in filename_list:
                #每个原点按照比例进行抽样
                rdd_tmp=sc.textFile(j)
                partitions_num=rdd_tmp.getNumPartitions()
                total_count=rdd_tmp.count()
                # sum_count=sum_count+total_count
                each_max_limit=max_sample_length/partitions_num
                eachpartitions_len=int(total_count/partitions_num*0.9)
                if  each_max_limit*pitch_length>eachpartitions_len:
                    print("每次抽样的样本总数需求不能大于ｒｄｄ的partition长度")
                    exit()
                rdd_tmp=rdd_tmp.mapPartitions(rdd_sample(eachpartitions_len,each_max_limit,pitch_length))#进行抽样,partition的顺序会被打乱,但是每个partition内部顺序不动
                # print("list':=",rdd_tmp.take(1))
                cz_rdd_list.append(rdd_tmp)
            all_rdd_list.append(sc.union(cz_rdd_list).repartition(1))
    else:
        print("一次输入的厂站-QFW数量必须和spark的worker数量一致")
        exit()
    return  all_rdd_list

#对福利也变换每个向量都作为变量
def cluster_FFT_file_to_rdd2(sc,filedir="/zd_data11.14/",filelist=[],work_num=4,fractions=0.50,max_sample_length=100,hdfs_addr="hdfs://sjfx1:9000",pitch_length=100):

    def rdd_sample(ep_len,max_length,pitch_len):#max_lengt每个partition抽取的个数
        import numpy as np
        import random
        from numpy import std
        #预先剔除2倍标准差以外的数据
        #在每个rdd的每个partition中按fractions的比例进行样本抽样
        #抽样比例：fractions,每个partiton的预估比例ep_len
        #时序模型需要，抽取连续的区间
        def map_func(iter):
            length=pitch_len*max_length#总抽样长度
            start_point=int((ep_len-length-1)*np.random.random())
            result=[]
            value=[]
            num=0
            j=0#pitch循环用
            for i in iter:
                if num>start_point and num<start_point+length+1:
                    if j%pitch_len==0 and j!=0:
                        one=np.array(value[1:value.__len__()])
                        w=[[e.real,e.imag] for e in np.fft.fft(one)]##取前面３个主要比重
                        out=[]
                        for n in w:
                            out.append(n[0])
                            out.append(n[1])
                        result.append([value[0],out])
                        value=[]
                        a=i.split(",")
                        value.append(str(a[0]))
                        value.append(float(a[1]))
                        j=1
                    else:
                        if j==0:
                            value=[]
                            a=i.split(",")
                            value.append(str(a[0]))
                            value.append(float(a[1]))
                            j=j+1
                        else:
                            a=i.split(",")
                            value.append(float(a[1]))
                            j=j+1
                num=num+1
            return result
        return map_func

    yd_num=list(filelist).__len__()
    print("本次处理点个数：=",yd_num)
    all_rdd_list=[]#所有点的list集合
    if(yd_num==work_num):#需要拟合的点数量正好等于work数量
        for i in filelist:
            cz_name=i[0]#厂站名字
            eq_type=i[1]#原点种类 F功率 Q电量 S风速
            file_length=float(i[3])
            filename_list=[hdfs_addr+filedir+"F"+str(eq_type)+"/"+str(e) for e in str(i[2]).split("|")]
            # print(filename_list)
            cz_rdd_list=[]#每个厂+Q，W，F
            sum_count=0
            for j in filename_list:
                #每个原点按照比例进行抽样
                rdd_tmp=sc.textFile(j)
                partitions_num=rdd_tmp.getNumPartitions()
                total_count=rdd_tmp.count()
                # sum_count=sum_count+total_count
                each_max_limit=max_sample_length/partitions_num
                eachpartitions_len=int(total_count/partitions_num*0.9)
                if  each_max_limit*pitch_length>eachpartitions_len:
                    print("每次抽样的样本总数需求不能大于ｒｄｄ的partition长度")
                    exit()
                rdd_tmp=rdd_tmp.mapPartitions(rdd_sample(eachpartitions_len,each_max_limit,pitch_length))#进行抽样,partition的顺序会被打乱,但是每个partition内部顺序不动
                # print("list':=",rdd_tmp.take(1))
                cz_rdd_list.append(rdd_tmp)
            all_rdd_list.append(sc.union(cz_rdd_list).repartition(1))
    else:
        print("一次输入的厂站-QFW数量必须和spark的worker数量一致")
        exit()
    return  all_rdd_list

#FFT的数据做相似度检验
def cluster_FFT_file_to_rdd(sc,filedir="/zd_data11.14/",filelist=[],work_num=4,fractions=0.50,max_sample_length=100,hdfs_addr="hdfs://sjfx1:9000",pitch_length=100):

    def rdd_sample(ep_len,max_length,pitch_len):#max_lengt每个partition抽取的个数
        import numpy as np
        import random
        from numpy import std
        #预先剔除2倍标准差以外的数据
        #在每个rdd的每个partition中按fractions的比例进行样本抽样
        #抽样比例：fractions,每个partiton的预估比例ep_len
        #时序模型需要，抽取连续的区间
        def map_func(iter):
            length=pitch_len*max_length#总抽样长度
            start_point=int((ep_len-length-1)*np.random.random())
            result=[]
            value=[]
            num=0
            j=0#pitch循环用
            for i in iter:
                if num>start_point and num<start_point+length+1:
                    if j%pitch_len==0 and j!=0:
                        one=np.array(value[1:value.__len__()])
                        vlaue_fft=np.fft.fft(one)
                        fft_value=np.sum(vlaue_fft[0:3]*[0.6,0.3,0.1])##取前面３个主要比重
                        result.append([value[0],fft_value.real,fft_value.imag])
                        value=[]
                        a=i.split(",")
                        value.append(str(a[0]))
                        value.append(float(a[1]))
                        j=0
                    else:
                        if j==0:
                            value=[]
                            a=i.split(",")
                            value.append(str(a[0]))
                            value.append(float(a[1]))
                            j=j+1
                        else:
                            a=i.split(",")
                            value.append(float(a[1]))
                            j=j+1
                num=num+1
            return result
        return map_func

    yd_num=list(filelist).__len__()
    print("本次处理点个数：=",yd_num)
    all_rdd_list=[]#所有点的list集合
    if(yd_num==work_num):#需要拟合的点数量正好等于work数量
        for i in filelist:
            cz_name=i[0]#厂站名字
            eq_type=i[1]#原点种类 F功率 Q电量 S风速
            file_length=float(i[3])
            filename_list=[hdfs_addr+filedir+"F"+str(eq_type)+"/"+str(e) for e in str(i[2]).split("|")]
            # print(filename_list)
            cz_rdd_list=[]#每个厂+Q，W，F
            sum_count=0
            for j in filename_list:
                #每个原点按照比例进行抽样
                rdd_tmp=sc.textFile(j)
                partitions_num=rdd_tmp.getNumPartitions()
                total_count=rdd_tmp.count()
                # sum_count=sum_count+total_count
                each_max_limit=max_sample_length/partitions_num
                eachpartitions_len=int(total_count/partitions_num*0.9)
                if  each_max_limit*pitch_length>eachpartitions_len:
                    print("每次抽样的样本总数需求不能大于ｒｄｄ的partition长度")
                    exit()
                rdd_tmp=rdd_tmp.mapPartitions(rdd_sample(eachpartitions_len,each_max_limit,pitch_length))#进行抽样,partition的顺序会被打乱,但是每个partition内部顺序不动
                # print("list':=",rdd_tmp.take(1))
                cz_rdd_list.append(rdd_tmp)
            all_rdd_list.append(sc.union(cz_rdd_list).repartition(1))
    else:
        print("一次输入的厂站-QFW数量必须和spark的worker数量一致")
        exit()
    return  all_rdd_list


#FFT和ｓｐｅａｒｍａｎ
#在一个ｒｄｄ中提取一段连续长短的数据
def cluster_FFT_spearman_to_rdd2(sc,filedir="/zd_data11.14/",
                                 filelist=[],work_num=4,
                                 hdfs_addr="hdfs://sjfx1:9000"
                                 ,start_point=50000,
                                 time_point="#",
                                 pitch_length=100000):

        def rdd_catch_pitch(sc,filename,length=200000,start_point=-1,time_point="#"):
            import numpy as np
            import re

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
                print("start_point:=",start_point)
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
                print("start_point:=",start_point)

            # 开始采样点
            def map_func_N(x):
                list_value=str((x[1])).split(",")
                return  [x[0],str(list_value[0]),float(list_value[1])]

            rdd=rdd.filter(lambda x:x[0]>=start_point and x[0]<start_point+length)
            rdd=rdd.map(map_func_N).persist()
            return rdd,time_point

        yd_num=list(filelist).__len__()
        print("本次处理点个数：=",yd_num)
        all_rdd_list=[]#所有点的list集合
        if(yd_num==work_num):#需要拟合的点数量正好等于work数量
            for i in filelist:
                cz_name=i[0]#厂站名字
                eq_type=i[1]#原点种类 F功率 Q电量 S风速
                file_length=float(i[3])
                filename_list=[hdfs_addr+filedir+"F"+str(eq_type)+"/"+str(e) for e in str(i[2]).split("|")]
                # print(filename_list)
                cz_rdd_list=[]#每个厂+Q，W，F
                sum_count=0
                time_x="#"
                for j in filename_list:
                    if(sum_count==0):
                      rdd_tmp,time_x=rdd_catch_pitch(sc,j,pitch_length,start_point=start_point,time_point=time_point)
                      cz_rdd_list.append(rdd_tmp)
                      sum_count=sum_count+1
                    else:
                      rdd_tmp,_=rdd_catch_pitch(sc,j,pitch_length,start_point=-1,time_point=time_x)
                      cz_rdd_list.append(rdd_tmp)
                      sum_count=sum_count+1
                all_rdd_list.append(sc.union(cz_rdd_list).repartition(1).sortBy(lambda x:[x[1],x[0]]).map(lambda x:[x[1],x[2]]))
        else:
            print("一次输入的厂站-QFW数量必须和spark的worker数量一致")
            exit()
        return  all_rdd_list


#厂站-QSW _数据集训练
# def inference_file_to_rdd(sc,filedir="/zd_data11.14/",filelist=[],work_num=4,hdfs_addr="hdfs://sjfx1:9000"):
#
#     def rdd_sample(fractions,ep_len,max_length):
#         import numpy as np
#         import random
#         #在每个rdd的每个partition中按fractions的比例进行样本抽样
#         #抽样比例：fractions,每个partiton的预估比例ep_len
#         #时序模型需要，抽取连续的区间
#         def map_func(iter):
#             start_point=ep_len*(1-fractions)*np.random.random()#开始取样点
#             length=fractions*ep_len#抽样长度
#             if length>max_length:#最大抽样长度
#                 length=max_length
#             result=[]
#             num=0
#             for i in iter:
#                 if num>start_point and num<start_point+length:
#                     value=str(i).split(",")
#                     result.append([str(value[0])+"|"+str(value[2]),float(value[1])])
#                 num=num+1
#             return result
#         return map_func
#
#     yd_num=list(filelist).__len__()
#     print("本次处理点个数：=",yd_num)
#     all_rdd_list=[]#所有点的list集合
#     if(yd_num==work_num):#需要拟合的点数量正好等于work数量
#         for i in filelist:
#             cz_name=i[0]#厂站名字
#             eq_type=i[1]#原点种类 F功率 Q电量 S风速
#             file_length=float(i[3])#文件长度
#             filename_list=i[2]#文件绝对名字
#             # print(filename_list)被抽样的文件
#             rdd_tmp=sc.textFile(hdfs_addr+filedir+"F"+str(eq_type)+"/"+str(filename_list))
#
#             #统计每个partition的长度，用于为每个元素编辑index
#             def func_count(num,iter):
#                 j=0
#                 for i in iter:
#                     j=j+1
#                 return [[num,j]]
#             each_p_list=rdd_tmp.mapPartitions(func_count).collect()#每个partition长度
#             each_p_list_count=list(each_p_count).__len__()#有几个partition
#             each_p_list_user=[]
#             if each_p_list_count!=1:#如果rdd的parttion>1，需要编辑index
#                for i in range(each_p_list_count):#累加
#                    if i==0:
#                       each_p_list_user.append(0)
#                    else:
#                       each_p_list_user.append(each_p_list_user[i-1]+each_p_list[i-1])
#             else:
#
#
#
#             def func_index(each_p_count):
#                 def map_func(num,iter):
#                     result=[]
#                     for i in iter:
#                         each_p_count[num]
#                     return result
#                 return map_func
#
#             rdd_tmp=rdd_tmp.mapPartitions(map_func).map(lambda x:[cz_name+"|"+eq_type,x])#进行抽样,partition的顺序会被打乱,但是每个partition内部顺序不动
#             all_rdd_list.append(rdd_tmp.repartition(1))
#     else:
#         print("一次输入的厂站-QFW数量必须和spark的worker数量一致")
#     return  all_rdd_list

#inference 准备数据集
def data_to_inference(addrs="sjfx1",port="50070",cz_FQW=[],network_num=4):
    from hdfs.client import Client #hdfs和本地文件的交互
    import pyhdfs as pd #判断文件是否存在
    import numpy as  np

    fs_pyhdfs = pd.HdfsClient(addrs,port)
    fs_hdfs = Client("http://"+addrs+":"+port)

    #全部样本 ['G_CFMY', 'Q', 'G_CFMY_1_001', 'G_CFMY_1_001FQ001.txt']
    inference_file_list=[]
    for i in list(cz_FQW):
        inference_file_list.extend([[i[0],i[1],e,fs_hdfs.status("/zd_data11.14/"+"F"+str(i[1])+"/"+str(e))['length']/np.power(1024,2)] for e in str(i[2]).split("|")])
    # print("处理前：",inference_file_list.__len__())
    # print(inference_file_list)
    #按network_num对齐文件数
    num=0
    yu_num=inference_file_list.__len__()%network_num
    # print("余数：=",yu_num)
    # print([inference_file_list[0]]*(network_num-inference_file_list.__len__()))
    inference_file_list.extend([inference_file_list[0]]*(network_num-yu_num))
    # print("处理后：",inference_file_list.__len__())
    from operator import itemgetter, attrgetter
    return sorted(inference_file_list,key=itemgetter(3))
    #['G_LYXGF', 'G_LYXGF_1_116NW001.1.txt', 0.41437244415283203]


#二、############################################################################
from dateutil import parser
#处理将不规范的日期调整成规范日期
spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
sc =spark.sparkContext
rdd=sc.textFile("hdfs://sjfx1:9000/zd_data11.14/FQ/G_CFMY_1_001FQ001.txt").map(lambda x:str(x).split(",")) \
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

