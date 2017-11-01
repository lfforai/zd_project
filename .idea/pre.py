from pyspark.conf import SparkConf
import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

from tensorflow.python.ops import variable_scope as vs

# from tensorflowonspark import TFCluster
# import pyspark.sql as sql_n       #spark.sql
# from pyspark import SparkContext  # pyspark.SparkContext dd
# from pyspark.conf import SparkConf #conf
#
# from pyspark.sql.types import *
# schema = StructType([
#     StructField("id",  StringType(), True),
#     StructField("value", FloatType(), True),
#     StructField("date", StringType(), True)]
# )
#
# os.environ['JAVA_HOME'] = "/tool_lf/java/jdk1.8.0_144/bin/java"
# os.environ["PYSPARK_PYTHON"] = "/root/anaconda3/bin/python"
# os.environ["HADOOP_USER_NAME"] = "root"
# conf=SparkConf().setMaster("spark://lf-MS-7976:7077")
# spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
# sc =spark.sparkContext
# sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)
#
# # 电量检查
# check="1"
# if(check=="0"):
#     # os.environ['JAVA_HOME'] = conf.get(SECTION, 'JAVA_HOME')
#     rd=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001.txt").map(lambda x:str(x).split(",")) \
#     .map(lambda x:[str(x[0]).replace("\'",""),x[1],str(x[2]).replace("\'","").lstrip()]) \
#     .map(lambda x:[str(x[0]).replace("[",""),float(x[1]),str(x[2]).replace("]","")])
#     df=sqlContext.createDataFrame(rd, "id:string,value:float,date:string")
#     df.createOrReplaceTempView("table1")
#     # df.filter(df.date=="2015-09-29 00:00:55").show()
#     list1=sqlContext.sql("select value from table1 where date between '2015-11-01 11:00:00' and '2015-11-01 11:09:59' ").rdd.map(list).collect()
#     print("a",list1)
#     value=sum(numpy.array(list1))
#     print("b",value)
#     print("d1",list1.__len__())
#
#     rd=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/G_CFYH_2_035FQ001.txt").map(lambda x:str(x).split(",")) \
#     .map(lambda x:[str(x[0]).replace("\'",""),x[1],str(x[2]).replace("\'","").lstrip()]) \
#     .map(lambda x:[str(x[0]).replace("[",""),float(x[1]),str(x[2]).replace("]","")])
#     df=sqlContext.createDataFrame(rd, "id:string,value:float,date:string")
#     df.createOrReplaceTempView("table1")
#     list2=sqlContext.sql("select value from table1 where date between '2015-11-01 11:00:00' and '2015-11-01 11:09:59' ").rdd.map(list).collect()
#     # print(list2)
#     list3=numpy.array(list2,dtype=float)[1:-1]
#     list4=numpy.array(list2,dtype=float)[0:list2.__len__()-2]
#     print("cat",list4-list3)
#     print("c",list2)
#     print("d",list2.__len__())
#
#     rd=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FW/G_CFYH_2_035FW001.txt").map(lambda x:str(x).split(",")) \
#     .map(lambda x:[str(x[0]).replace("\'",""),x[1],str(x[2]).replace("\'","").lstrip()]) \
#     .map(lambda x:[str(x[0]).replace("[",""),float(x[1]),str(x[2]).replace("]","")])
#     df=sqlContext.createDataFrame(rd, "id:string,value:float,date:string")
#     df.createOrReplaceTempView("table2")
#     list1=sqlContext.sql("select * from table2 where date between '2015-11-01 11:00:00' and '2015-11-01 11:09:59' ").rdd.map(list).collect()
#     print(numpy.average(numpy.array(numpy.array(list1)[:,1],dtype=float)*60*10/3600))
#
# # 电量增量检查
# check="1"
# if(check=="0"):
#     rd=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001.txt").map(lambda x:str(x).split(",")) \
#         .map(lambda x:[str(x[0]).replace("\'",""),x[1],str(x[2]).replace("\'","").lstrip()]) \
#         .map(lambda x:[str(x[0]).replace("[",""),float(x[1]),str(x[2]).replace("]","")])
#     df=sqlContext.createDataFrame(rd, "id:string,value:float,date:string")
#     df.createOrReplaceTempView("table1")
#     # df.filter(df.date=="2015-09-29 00:00:55").show()
#     # list1=sqlContext.sql("select max(value),min(value) from table1")
#     # list1.show()
#     # print(df.count())
#     # print(df.filter("value>1000").count())
#     import pyhdfs as pd
#     fs = pd.HdfsClient("127.0.0.1", 9000)
#     if(not fs.exists("/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000.txt")):
#         num_list=10
#         def fuc(iterator):
#             value_list=[]
#             num=0
#             value=''
#             for i in iterator:
#                 if num%num_list==0:
#                     if(value==''):
#                         value=value+str(i)
#                         num=num+1
#                     else:
#                         value_list.append(value)
#                         value=str(i)
#                         num=1
#                 else:
#                     value=value+','+str(i)
#                     num=num+1
#             return value_list
#         df.filter("value<1000").filter("value>0").select("value")\
#            .rdd.map(list).map(lambda x:str(x).replace("[","").replace("]","")).mapPartitions(fuc)\
#             .saveAsTextFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000.txt")
#
#     print(sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000.txt").take(10))
#     print(sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000.txt").count())

# 对抗神经网建模
print("---------------------------------------------------------------")
check="0"
if(check=="0"):
    tf.reset_default_graph()
    import matplotlib.pyplot as plt
    import seaborn as sns # for pretty plots
    from scipy.stats import norm
    import pyhdfs as pd
    import numpy as np
    fs = pd.HdfsClient("127.0.0.1", 9000)
    #[filename]=fs.walk("/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000_30.txt/")
    [filename]=fs.walk("/lf/")
    files_list=list(filename)
    files_local=[item for item in map(lambda x:str("hdfs://127.0.0.1:9000"+files_list[0])+str(x),list(files_list[2])[1:])]
    print(files_local)
# //对抗神经网拟合
#神经网构成
#  MLP - used for D_pre, D1, D2, G networks
    M=30 # minibatch size
    pitch=100
with tf.variable_scope("D1", reuse=tf.AUTO_REUSE):
    # construct learnable parameters within local scope
    w11=tf.get_variable("w10", [30, 50])*0.2
    b11=tf.get_variable("b10", [50])*0.2
    w21=tf.get_variable("w11", [50, 30])*0.2
    b21=tf.get_variable("b11", [30])*0.2
    w31=tf.get_variable("w12", [30, 15])*0.2
    b31=tf.get_variable("b12", [15])*0.2
    w41=tf.get_variable("w13", [15,1])*0.2
    b41=tf.get_variable("b13", [1])*0.2

    def mlp_D1(input):
            # construct learnable parameters within local scope
            # w11=tf.get_variable("w10", [input.get_shape()[1], 150], initializer=tf.random_normal_initializer())
            # b11=tf.get_variable("b10", [150], initializer=tf.constant_initializer(0.0))
            # w21=tf.get_variable("w11", [150, 70], initializer=tf.random_normal_initializer())
            # b21=tf.get_variable("b11", [70], initializer=tf.constant_initializer(0.0))
            # w31=tf.get_variable("w12", [70, 35], initializer=tf.random_normal_initializer())
            # b31=tf.get_variable("b12", [35], initializer=tf.constant_initializer(0.0))
            # w41=tf.get_variable("w13", [35,1], initializer=tf.random_normal_initializer())
            # b41=tf.get_variable("b13", [1], initializer=tf.constant_initializer(0.0))

            fc11=tf.nn.tanh(tf.matmul(input,w11)+b11)
            fc11 = tf.nn.dropout(fc11, keep_prob=1.0)
            fc12=tf.nn.tanh(tf.matmul(fc11,w21)+b21)
            fc12 = tf.nn.dropout(fc12, keep_prob=1.0)
            fc13=tf.nn.tanh(tf.matmul(fc12,w31)+b31)
            fc13 = tf.nn.dropout(fc13, keep_prob=1.0)
            fc14=tf.nn.tanh(tf.matmul(fc13,w41)+b41)
            return fc14, [w11,b11,w21,b21,w31,b31,w41,b41]

    # D(x)
    x_node=tf.placeholder(dtype=tf.float32, shape=(None,M))
    # x_node=tf.placeholder(tf.float32, shape=(None,M)) # input M normally distributed floats
    fc1,theta_d=mlp_D1(x_node) # output likelihood of being normally distributed
    D1=tf.maximum(tf.minimum(fc1,.99), 0.01) # clamp as a probability


    # with tf.device('/cpu:0'):
    # prepair data--------------------------------------------------------------------------------------
    def read_data(file_queue):
        reader = tf.TextLineReader()
        key, value = reader.read(file_queue)
        defaults = [[0.0]]*M
        # print(defaults)
        list_value = tf.decode_csv(value, defaults)
        list_value_tensor=tf.stack(list_value)
        #因为使用的是鸢尾花数据集，这里需要对y值做转换
        return list_value_tensor

    def create_pipeline(filename, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer(filename,num_epochs=num_epochs)
        example= read_data(file_queue)
        min_after_dequeue = 2000
        capacity = min_after_dequeue + batch_size
        example_batch= tf.train.shuffle_batch(
            [example], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue
        )
        # print(example_batch)
        return example_batch

    # 开始训练
    x_train_batch= create_pipeline(files_local, pitch)


    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, "/tool_lf/lf/model-17000.ckpt")
        # sess.run(local_init_op)
        # print(np.reshape(np.random.random(pitch*M),(pitch,M)))
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("开始输出：")
        int_num=0
        print(sess.run(theta_d[0]))
        print(sess.run(w11))

        try:
            #while not coord.should_stop():
            while True:
                # x=np.reshape(np.ones(M*pitch)*100000+np.random.normal(size=M*pitch)*2,(pitch,M))
                # print(x)
                # x=np.reshape(np.ones(M*pitch)*999+np.abs(np.random.normal(size=pitch*M)*1000000),(pitch,M))
                x=sess.run(x_train_batch)#sampled m-batch from zd_data
                # print(x)
                x=np.sort(x,axis=1)
                # print(x)
                print("---------------")
                print(sess.run(D1,feed_dict={x_node:x}))
                int_num=int_num+1
                if int_num==2:
                    break
        except tf.errors.OutOfRangeError:
            print ('Done reading')
        finally:
            coord.request_stop()
            coord.join(threads)
        # save_path = saver.save(sess,"/tool_lf/lf/model-last.ckpt")
        sess.close()
        print ("--ok！--")