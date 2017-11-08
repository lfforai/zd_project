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
conf=SparkConf().setMaster("spark://192.168.1.70:7077")
# os.environ['JAVA_HOME'] = conf.get(SECTION, 'JAVA_HOME')
spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
sc =spark.sparkContext
sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

# 放置文件
def bl_file(file_dir="/data_zd/data/"):
    [files]=os.walk(file_dir)
    files_list=list(files)
    list_fl={"FS":"FS","FQ":"FQ","FW":"FW"}
    files_local=[item for item in map(lambda x:str(file_dir)+str(x),files_list[2])]
    files_name=[item for item in files_list[2]]
    print("file_loacl:=",files_local)
    print("file_name:=",files_name)
    return files_local,files_name  #返回本地文件名字（路径）和不带路进

# 路径
import multiprocessing
import time
import pyhdfs as pd
fs = pd.HdfsClient("192.168.1.67", 8020)

def local_to_hdfs(hdfs_path="hdfs://192.168.1.67:8020/zd_data2/",local_filename="",file_hdfsname=""):
    if(file_hdfsname.__contains__("FS")):
        mid_path="FS/"
    else:
        if(file_hdfsname.__contains__("FQ")):
            mid_path="FQ/"
        else:
            mid_path="FW/"
    if(not fs.exists("/zd_data2/"+mid_path+file_hdfsname)):
        print("存储开始："+hdfs_path+mid_path+file_hdfsname)
        print(local_filename)
        rdd=sc.textFile("file://"+local_filename).repartition(1).saveAsTextFile(hdfs_path+mid_path+file_hdfsname)
        print("存储结束："+hdfs_path+mid_path+file_hdfsname)
    else:
        print("/zd_data2/"+mid_path+file_hdfsname+"exsit!")

        #     .map(lambda x:str(x).split(","))\
        # .map(lambda x:(int(x[0]),float(x[1])))
        #         spark.createDataFrame(rdd, schema).sort("id","value","data").show(100)
        #images = sc.newAPIHadoopFile(args.images, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
        #                              keyClass="org.apache.hadoop.io.BytesWritable",
        #                              valueClass="org.apache.hadoop.io.NullWritable")

# 按小时，分钟，加工数据
def sub_each(file_hdfsname_all=" ",
             ouput_root="hdfs://192.168.1.67:8020",
             old_fold="/zd_data2/",
             new_fold="idea_ok/",
             filename="",
             time_fre="hour"):

    rdda = sc.parallelize([["a",0.5,"b"],["a",0.5,"b"]])
    result=spark.createDataFrame(rdda,"id:string,value:float,date:string")
    if(file_hdfsname_all.__contains__("FS")):
        mid_path="FS/"
    else:
        if(file_hdfsname_all.__contains__("FQ")):
            mid_path="FQ/"
        else:
            mid_path="FW/"

    def fuc(iterator):
        value_list=[]
        num=0
        pre=0
        for i in iterator:
            if num==0:
                pre=i
                num=num+1
            else:
                hou=i
                value_list.append([str(hou[0]),float(hou[1])-float(pre[1]),str(hou[2])])
                pre=hou
        return value_list

    # map(lambda x:str(x).split(","))
    if(mid_path.__eq__("FQ/")):
        print("FQ/")
        print("开始替换日期并前后项做差："+file_hdfsname_all)
        # print(sc.textFile(file_hdfsname_all).map(lambda x:list(str(x).split(","))).first())
        if(fs.exists(old_fold+mid_path+new_fold+filename)):
            print("文件已有了")
            # print(sc.textFile("hdfs://192.168.1.67:8020/zd_data2/FQ/new/1.txt").first())
            rdd=sc.textFile(ouput_root+old_fold+mid_path+new_fold+filename).map(lambda x:str(x).split(",")) \
                .map(lambda x:[str(x[0]).replace("\'",""),x[1],str(x[2]).replace("\'","").lstrip()]) \
                .map(lambda x:[str(x[0]).replace("[",""),float(x[1]),str(x[2]).replace("]","")])
            # print(rdd.first())
            # 用于排序的函数
            df=sqlContext.createDataFrame(rdd, "id:string,value:float,date:string")
            from pyspark.sql.types import StringType
            if(time_fre=="hour"):
             sqlContext.udf.register("Subdate",lambda x:x[0:13],StringType())
             df.createOrReplaceTempView("table1")
             result=sqlContext.sql("SELECT sum(value) as sum_n,Subdate(date) as date_q from table1 where value>0 and value <100000 group by date_q")
             return result
            else:
               if(time_fre=="10minute"):
                 sqlContext.udf.register("Subdate",lambda x:x[0:15],StringType())
                 df.createOrReplaceTempView("table1")
                 result=sqlContext.sql("SELECT sum(value) as sum_n,Subdate(date) as date_q from table1 where value>0 and value <100000 group by date_q")
                 return result
               else:
                   if(time_fre=="1minute"):
                       sqlContext.udf.register("Subdate",lambda x:x[0:16],StringType())
                       df.createOrReplaceTempView("table1")
                       result=sqlContext.sql("SELECT sum(value) as sum_n,Subdate(date) as date_q from table1 where value>0 and value <100000 group by date_q")
                       return result
        else:
            print("开始保存文件")
            # .sortBy(lambda x: (float(x[1]),str(x[2])))
            sc.textFile(file_hdfsname_all,1).map(lambda x:list(str(x).split(","))).filter(lambda x:x.__len__()==3).mapPartitions(fuc)\
                .saveAsTextFile(ouput_root+old_fold+mid_path+new_fold+filename)
            rdd=sc.textFile(ouput_root+old_fold+mid_path+new_fold+filename).map(lambda x:str(x).split(",")) \
                .map(lambda x:[str(x[0]).replace("\'",""),x[1],str(x[2]).replace("\'","").lstrip()]) \
                .map(lambda x:[str(x[0]).replace("[",""),float(x[1]),str(x[2]).replace("]","")])
            #.map(lambda x:x[1]).filter(lambda x:True if str(x).__eq__("'") else False).count())
            # map(lambda x:float(str(x[1]).replace("'","0.0"))).filter(lambda x:False if str(x).__eq__("'")  else True).first())
            # .map(lambda x:[x[0].replace("\'","").replace("\'",""),float(x[1].replace("\'","").replace("\'","")),x[2].replace("\'","").replace("\'","")])
            # 用于排序的函数
            df=sqlContext.createDataFrame(rdd, "id:string,value:float,date:string")
            from pyspark.sql.types import StringType
            if(time_fre=="hour"):
                sqlContext.udf.register("Subdate",lambda x:x[0:13],StringType())
                df.createOrReplaceTempView("table1")
                result=sqlContext.sql("SELECT sum(value) as sum_n,Subdate(date) as date_q from table1 where value>0 and value <100000 group by date_q")
                return result
            else:
                if(time_fre=="10minute"):
                    sqlContext.udf.register("Subdate",lambda x:x[0:15],StringType())
                    df.createOrReplaceTempView("table1")
                    result=sqlContext.sql("SELECT sum(value) as sum_n,Subdate(date) as date_q from table1 where value>0 and value <100000 group by date_q")
                    return result
                else:
                    if(time_fre=="1minute"):
                        sqlContext.udf.register("Subdate",lambda x:x[0:16],StringType())
                        df.createOrReplaceTempView("table1")
                        result=sqlContext.sql("SELECT sum(value) as sum_n,Subdate(date) as date_q from table1 where value>0 and value <100000 group by date_q")
                        return result
        print("替换日期并前后项做差结束："+file_hdfsname_all)

    # 功率,求均值
    if(mid_path.__eq__("FW/")):
            print("FW/")
            rdd=sc.textFile(file_hdfsname_all).map(lambda x:list(str(x).split(","))).filter(lambda x:x.__len__()==3) \
                .map(lambda x:[str(x[0]).replace("\'",""),float(x[1]),str(x[2]).replace("\'","").lstrip()])
            df=sqlContext.createDataFrame(rdd, "id:string,value:float,date:string")
            df.createOrReplaceTempView("table2")
            from pyspark.sql.types import StringType
            if(time_fre=="hour"):
                    sqlContext.udf.register("Subdate",lambda x:x[0:13],StringType())
                    result=sqlContext.sql("SELECT avg(value) as avg_n,Subdate(date) as date_w from table2 group by date_w")
                    print("ok!")
                    return result
            else:
                if(time_fre=="10minute"):
                    sqlContext.udf.register("Subdate",lambda x:x[0:15],StringType())
                    result=sqlContext.sql("SELECT avg(value)*(10*60)/3600 as avg_n,Subdate(date) as date_w from table2 group by date_w")
                    print("ok!")
                    return result
                else:
                    if(time_fre=="1minute"):
                        sqlContext.udf.register("Subdate",lambda x:x[0:16],StringType())
                        result=sqlContext.sql("SELECT avg(value)*60/3600 as avg_n,Subdate(date) as date_w from table2 group by date_w")
                        print("ok!")
                        return result

    if(mid_path.__eq__("FS/")):
            print("FS/")
            rdd=sc.textFile(file_hdfsname_all).map(lambda x:list(str(x).split(","))).filter(lambda x:x.__len__()==3) \
                .map(lambda x:[str(x[0]).replace("\'",""),float(x[1]),str(x[2]).replace("\'","").lstrip()])
            df=sqlContext.createDataFrame(rdd, "id:string,value:float,date:string")
            df.createOrReplaceTempView("table2")
            from pyspark.sql.types import StringType
            if(time_fre=="hour"):
                sqlContext.udf.register("Subdate",lambda x:x[0:13],StringType())
                result=sqlContext.sql("SELECT avg(value) as avg_f,Subdate(date) as date_f from table2 group by date_f")
                print("ok!")
                return result
            else:
                if(time_fre=="10minute"):
                    sqlContext.udf.register("Subdate",lambda x:x[0:15],StringType())
                    result=sqlContext.sql("SELECT avg(value) as avg_f,Subdate(date) as date_f from table2 group by date_f")
                    print("ok!")
                    return result
                else:
                    if(time_fre=="1minute"):
                        sqlContext.udf.register("Subdate",lambda x:x[0:16],StringType())
                        result=sqlContext.sql("SELECT avg(value) as avg_f,Subdate(date) as date_f from table2 group by date_f")
                        print("ok!")
                        return result


if __name__ == "__main__":
    # 数据导入
    local,hdfs =bl_file()
    pool = multiprocessing.Pool(processes = 3)
    print(local)
    print(hdfs)
    result=[]
    print(len(local))
    for i  in range(len(local)):
       local_to_hdfs("hdfs://192.168.1.67:8020/zd_data2/",local[i],hdfs[i])
    print("Sub-process(es) done.")

    # 提取小时数据
    filenum="35"
    # fs.delete("/zd_data2/rezult/"+filenum+".txt")
    if(not fs.exists("/zd_data2/rezult/"+filenum+".txt")):
        FQ=sub_each("hdfs://192.168.1.67:8020/zd_data2/FQ/G_CFYH_2_0"+filenum+"FQ001.txt",filename="G_CFYH_2_0"+filenum+"FQ001.txt",time_fre="10minute")
        FQ.show(10)
        FW=sub_each("hdfs://192.168.1.67:8020/zd_data2/FW/G_CFYH_2_0"+filenum+"FW001.txt",filename="G_CFYH_2_0"+filenum+"FW001.txt",time_fre="10minute")
        FW.show(10)
        FS=sub_each("hdfs://192.168.1.67:8020/zd_data2/FS/G_CFYH_2_0"+filenum+"FS001.txt",filename="G_CFYH_2_0"+filenum+"FS001.txt",time_fre="10minute")
        FS.show(10)
        FQ.createOrReplaceTempView("FQ")
        FW.createOrReplaceTempView("FW")
        FS.createOrReplaceTempView("FS")
        sqlContext.sql("SELECT date_q,sum_n,avg_n,avg_f from FQ,FW,FS where date_w=date_q and date_q=date_f order by date_q") \
            .repartition(1).write.mode("overwrite").parquet("hdfs://192.168.1.67:8020/zd_data2/rezult/"+filenum+".txt")

    df=sqlContext.read.parquet("hdfs://192.168.1.67:8020/zd_data2/rezult/"+filenum+".txt")
    df.createOrReplaceTempView("rezult") #        date|sub|rato|FQ|FW
    df=sqlContext.sql("select date_q as date,sum_n-avg_n as sub,(sum_n-avg_n)/sum_n*100 as rato,sum_n as FQ,avg_n as FW,avg_f as FS from rezult order by date_q")
    df.createOrReplaceTempView("rezult")
    print("data like :")
    df.show()
    print("cout like :")
    df.count()
    #第一次标准差全样
    times=1.5
    var_sub=round(numpy.power(df.cov("sub","sub"),0.5),2)
    var_rato=round(numpy.power(df.cov("rato","rato"),0.5),2)
    sub_rato=df.selectExpr("avg(sub)","avg(rato)").collect()
    avg_sub=sub_rato[0][0]
    avg_rato=sub_rato[0][1]
    max_range_sub=avg_sub+times*var_sub
    min_range_sub=avg_sub-times*var_sub
    max_range_rato=avg_rato+times*var_rato
    min_range_rato=avg_rato-times*var_rato
    print("第一次差值方差：",var_sub)
    print("第一次比率方差：",var_rato)
    print("第一次差值均值：",avg_sub)
    print("第一次比率均值：",avg_rato)
    print(min_range_sub)
    print(max_range_sub)
    print(min_range_rato)
    print(max_range_rato)

    print("first---------------------")
    #第二次标准差样本
    # from pyspark.sql.types import FloatType
    # sqlContext.registerFunction("eq_sub_min",lambda x:min_range_sub,FloatType())
    # from pyspark.sql.types import FloatType
    # sqlContext.registerFunction("eq_sub_max",lambda x:max_range_sub,FloatType())
    # from pyspark.sql.types import FloatType
    # sqlContext.registerFunction("eq_rato_min",lambda x:min_range_rato,FloatType())
    # from pyspark.sql.types import FloatType
    # sqlContext.registerFunction("eq_rato_max",lambda x:max_range_rato,FloatType())
    se=sqlContext.sql("select sub,rato from rezult where sub <"+str(max_range_sub)+" and sub >"+str(min_range_sub)+" and rato >"+str(min_range_rato)+" and rato <"+str(max_range_rato))
    var_sub=round(numpy.power(se.cov("sub","sub"),0.5),2)
    var_rato=round(numpy.power(se.cov("rato","rato"),0.5),2)
    sub_rato=se.selectExpr("avg(sub)","avg(rato)").collect()
    avg_sub=round(sub_rato[0][0],0)
    avg_rato=round(sub_rato[0][1],0)
    max_range_sub=avg_sub+times*var_sub
    min_range_sub=avg_sub-times*var_sub
    max_range_rato=avg_rato+times*var_rato
    min_range_rato=avg_rato-times*var_rato
    print("第二次差值方差：",var_sub)
    print("第二次比率方差：",var_rato)
    print("第二次差值均值：",avg_sub)
    print("第二次比率均值：",avg_rato)
    print(min_range_sub)
    print(max_range_sub)
    print(min_range_rato)
    print(max_range_rato)
    print("second---------------------")

    def pice(a,filenpath,num=24,div=3):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        # 必须配置中文字体，否则会显示成方块
        # 注意所有希望图表显示的中文必须为unicode格式
        zhfont1 = mpl.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
        font_size = 7 # 字体大小
        fig_size = (16, 8) # 图表大小
        names = (u'绝对差值', u'差异百分比') # 姓名
        # subjects = date # 科目
        # scores = (tuple(float(sub)), tuple(float(rato))) # 成绩

        # 更新字体大小
        mpl.rcParams['font.size'] = font_size
        # 更新图表大小
        mpl.rcParams['figure.figsize'] = fig_size
        # 设置柱形图宽度
        bar_width = 0.2

        index = np.arange(len(a[:,0]))

        if np.average(a[:,1].astype(np.float),axis=0)<min_range_sub or np.average(a[:,1].astype(np.float),axis=0)>max_range_sub:
            warn_sub="绝对差数据异常预警，请查询！"
        else:
            warn_sub=str("绝对差数据正常，在全样本")+str(div)+str("倍标准差之内")

            # 绘制「小明」的成绩
        rects1 = plt.bar(index,a[:,1], bar_width, color='#0072BC', label=names[0])

        # 绘制「小红」的成绩
        if np.average(a[:,2].astype(np.float),axis=0)<min_range_rato or np.average(a[:,2].astype(np.float),axis=0)>max_range_rato:
            warn_rato="相对差数据异常预警，请查询！"
        else:
            warn_rato=str("相对差数据正常,在全样本")+str(div)+str("倍标准差之内")

        rects2 = plt.bar(index + bar_width,a[:,2] , bar_width, color='#ED1C24', label=names[1])
        # X轴标题
        plt.xticks(index + bar_width, range(num), fontproperties=zhfont1)
        # Y轴范围
        plt.ylim(ymax=1000, ymin=-200)
        # 图表标题

        plt.title('两年—对差平均值:'+str(avg_sub)+' 方差:'+str(var_sub)+"||"+' 比差平均值:'+str(avg_rato)+' 方差:'+str(var_rato)
                  +"\n"+warn_sub+"\n"+warn_rato+"\n"+"样本时间区间"+a[:,0][0]+"——"+a[:,0][num-1], fontproperties=zhfont1)
        # 图例显示在图表下方
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5, prop=zhfont1)

        # 添加数据标签
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
                # 柱形图边缘用白色填充，纯粹为了美观
                rect.set_edgecolor('white')

        add_labels(rects1)
        add_labels(rects2)
        # 图表输出到本地
        # plt.show()
        plt.savefig("/lf/"+filenpath+"/"+str(a[:,0][0])+"—"+str(a[:,0][num-1])+'.png')
        plt.clf()
        return a[:,0][num-1]

    start="2013-03-03 00:00:00"
    i=1
    while True:
        # try:
        a=[]
        sub=sqlContext.sql("select date,sub,rato from  rezult where date >"+str("\'"+start+"\'")).rdd.map(list).map(lambda x:[x[0],round(x[1],2),round(x[2],2)]).take(10)
        a=numpy.array(sub)
        start=pice(a,filenum,num=10,div=2)
        # except:
        #   break

        # 必须配置中文字体，否则会显示成方块
        # 注意所有希望图表显示的中文必须为unicode格式