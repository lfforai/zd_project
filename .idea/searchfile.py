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
# spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
# sc =spark.sparkContext
# sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

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
# check="0"
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
#     if(not fs.exists("/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000_20_1.txt")):
#         num_list=20
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
#         df.filter("value<100").filter("value>0").select("value")\
#            .rdd.map(list).map(lambda x:str(x).replace("[","").replace("]","")).mapPartitions(fuc)\
#             .saveAsTextFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000_20_1.txt")

    # print(sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000_30.txt").take(10))
    # print(sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000_30.txt").count())

# 对抗神经网建模
print("---------------------------------------------------------------")
check="0"
if(check=="0"):

    import matplotlib.pyplot as plt
    import seaborn as sns # for pretty plots
    from scipy.stats import norm
    import pyhdfs as pd
    import numpy as np
    fs = pd.HdfsClient("127.0.0.1", 9000)
    [filename]=fs.walk("/zd_data2/FQ/idea_ok/G_CFYH_2_035FQ001_1000_20_1.txt/")
    files_list=list(filename)
    files_local=[item for item in map(lambda x:str("hdfs://127.0.0.1:9000"+files_list[0])+str(x),list(files_list[2])[1:])]
    print(files_local)

# //对抗神经网拟合
#神经网构成
#  MLP - used for D_pre, D1, D2, G networks
# with tf.device("/gpu:0"):
    M=20 # minibatch size
    pitch=200
    def mlp(input, output_dim):
        # construct learnable parameters within local scope
        w1=tf.get_variable("w0", [input.get_shape()[1], 300], initializer=tf.random_normal_initializer())
        b1=tf.get_variable("b0", [300], initializer=tf.constant_initializer(0.0))
        w2=tf.get_variable("w1", [300, 150], initializer=tf.random_normal_initializer())
        b2=tf.get_variable("b1", [150], initializer=tf.constant_initializer(0.0))
        w3=tf.get_variable("w2", [150, 75], initializer=tf.random_normal_initializer())
        b3=tf.get_variable("b2", [75], initializer=tf.constant_initializer(0.0))
        w4=tf.get_variable("w3", [75,output_dim], initializer=tf.random_normal_initializer())
        b4=tf.get_variable("b3", [output_dim], initializer=tf.constant_initializer(0.0))
        # nn operators
        fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
        fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
        fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
        fc2 = tf.nn.dropout(fc2, keep_prob=0.5)
        fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
        fc3 = tf.nn.dropout(fc3, keep_prob=0.5)
        fc4=tf.nn.tanh(tf.matmul(fc3,w4)+b4)
        return fc4, [w1,b1,w2,b2,w3,b3,w4,b4]

    def mlp_D1(input):
        # construct learnable parameters within local scope
        w11=tf.get_variable("w10", [input.get_shape()[1], 20], initializer=tf.random_normal_initializer())
        b11=tf.get_variable("b10", [20], initializer=tf.constant_initializer(0.0))
        w21=tf.get_variable("w11", [20, 10], initializer=tf.random_normal_initializer())
        b21=tf.get_variable("b11", [10], initializer=tf.constant_initializer(0.0))
        w31=tf.get_variable("w12", [10, 5], initializer=tf.random_normal_initializer())
        b31=tf.get_variable("b12", [5], initializer=tf.constant_initializer(0.0))
        w41=tf.get_variable("w13", [5,1], initializer=tf.random_normal_initializer())
        b41=tf.get_variable("b13", [1], initializer=tf.constant_initializer(0.0))
        # nn operators
        fc11=tf.nn.tanh(tf.matmul(input,w11)+b11)
        fc11 = tf.nn.dropout(fc11, keep_prob=0.5)
        fc12=tf.nn.tanh(tf.matmul(fc11,w21)+b21)
        fc12 = tf.nn.dropout(fc12, keep_prob=0.5)
        fc13=tf.nn.tanh(tf.matmul(fc12,w31)+b31)
        fc13 = tf.nn.dropout(fc13, keep_prob=0.5)
        fc14=tf.nn.tanh(tf.matmul(fc13,w41)+b41)
        return fc14, [w11,b11,w21,b21,w31,b31,w41,b41]

        # re-used for optimizing all networks
    def momentum_optimizer(loss,var_list):
            batch = tf.Variable(0)
            # learning_rate = tf.train.exponential_decay(
            #     0.001,                # Base learning rate.
            #     batch,  # Current index into the dataset.
            #     TRAIN_ITERS // 4,          # Decay step - this decays 4 times throughout training process.
            #     0.95,                # Decay rate.
            #     staircase=True)
            #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
            optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss,global_step=batch,var_list=var_list)
            return optimizer

    with tf.variable_scope("G"):
        z_node=tf.placeholder(dtype=tf.float32, shape=(None,M))
        # print(z_node)
        G,theta_g=mlp(z_node,M) # generate normal transformation of Z
        G=tf.multiply(100.0,G) # scale up by 20 to match range

    with tf.variable_scope("D") as scope:
        # D(x)
        x_node=tf.placeholder(dtype=tf.float32, shape=(None,M))
        # x_node=tf.placeholder(tf.float32, shape=(None,M)) # input M normally distributed floats
        fc1,theta_d=mlp_D1(x_node) # output likelihood of being normally distributed
        D1=tf.maximum(tf.minimum(fc1,.99), 0.01) # clamp as a probability
        tf.summary.scalar('D1', tf.reduce_mean(D1))
        # make a copy of D that uses the same variables, but takes in G as input
        scope.reuse_variables()
        fc2,theta_d=mlp_D1(G)
        D2=tf.maximum(tf.minimum(fc2,.99), 0.01)
        tf.summary.scalar('D2', tf.reduce_mean(D2))

    obj_d=tf.reduce_mean(tf.add(tf.log(D1),tf.log(float(1)-D2)))
    obj_g=tf.reduce_mean(tf.log(D2))

# with tf.device("/cpu:0"):
    # set up optimizer for G,D
    opt_d=momentum_optimizer(1-obj_d, theta_d)
    tf.summary.scalar('opt_d',obj_d)
    opt_g=momentum_optimizer(1-obj_g, theta_g) # maximize log(D(G(z)))
    tf.summary.scalar('opt_g',obj_g)

    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    # saver = tf.train.Saver()
    saver = tf.train.Saver()
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
        min_after_dequeue = 1000
        capacity = min_after_dequeue + batch_size
        example_batch= tf.train.shuffle_batch(
            [example], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue
        )
        # print(example_batch)
        return example_batch

    # 开始训练
    x_train_batch= create_pipeline(files_local, pitch)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    merged_summary_op = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
            sess.run(init_op)
            sess.run(local_init_op)
            # print(np.reshape(np.random.random(pitch*M),(pitch,M)))
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            int_num=0
            summary_writer = tf.summary.FileWriter('/tmp/mnist_logs', sess.graph)
            # print(sess.run(x_train_batch))
            # Retrieve a single instance:
            try:
                #while not coord.should_stop():
                while True:
                    for j in range(1):
                      x=np.array(sess.run(x_train_batch))
                      x=np.sort(x,axis=1)
                      # sampled m-batch from zd_dat
                      z=np.ones(shape=(M*pitch),dtype=float)*6+np.random.normal(size=M*pitch)
                      z=np.sort(np.reshape(z,(pitch,M)),axis=1)
                      #z=np.linspace(-500,500,M*pitch,dtype=float)+np.random.normal(size=M*pitch)*0.01
                      sess.run(opt_d,feed_dict={x_node:x,z_node:z})
                    for m in range(5):
                      #z=np.linspace(-500,500,M*pitch,dtype=float)+np.random.normal(size=M*pitch)*0.01
                      z=np.ones(shape=(M*pitch),dtype=float)*6+np.random.normal(size=M*pitch)
                      z=np.sort(np.reshape(z,(pitch,M)),axis=1)
                      sess.run(opt_g,feed_dict={z_node:z})
                    int_num=int_num+1
                    if(int_num>150000):
                      break
                    if(int_num%5==0):
                      rezult=sess.run(merged_summary_op,feed_dict={x_node:x,z_node:np.reshape(z,(pitch,M))})
                      summary_writer.add_summary(rezult, int_num)
                    if(int_num%1000==0 and int_num!=0):
                        save_path = saver.save(sess,"/tool_lf/lf/model-"+str(int_num)+".ckpt")
                    # print (example)
            except tf.errors.OutOfRangeError:
                print ('Done reading')
            finally:
                coord.request_stop()
                coord.join(threads)
            save_path = saver.save(sess,"/tool_lf/lf/model-last.ckpt")
            sess.close()
            print ("--ok！--")


