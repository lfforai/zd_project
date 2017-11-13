#///////////////////////////////////////////////////////////////////////////////////////

#./spark-submit --conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" --conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4"  ~/IdeaProjects/pyspark_t/.idea/spark-warehouse/spark-streaming/exp.py



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
#
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from hdfs import *
client_N = Client("http://127.0.0.1:50070")

import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

from tensorflowonspark import TFCluster
import AR_model_use
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader
# import pyspark.sql as sql_n       #spark.sql
# from pyspark import SparkContext  # pyspark.SparkContext dd
# from pyspark.conf import SparkConf #conf

os.environ['JAVA_HOME'] = "/tool_lf/java/jdk1.8.0_144/bin/java"
os.environ["PYSPARK_PYTHON"] = "/root/anaconda3/bin/python"
os.environ["HADOOP_USER_NAME"] = "root"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["spark.executorEnv.LD_LIBRARY_PATH"]="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"
os.environ["spark.executorEnv.CLASSPATH"]="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}"
os.environ["spark.executorEnv.HADOOP_HDFS_HOME"]="/tool_lf/hadoop/hadoop-2.7.4"

def get_available_gpus_len():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU'].__len__()

def map_fun(args, ctx):
    from tensorflowonspark import TFNode
    from datetime import datetime
    import math
    import numpy
    import tensorflow as tf
    import time

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    cluster_spec = ctx.cluster_spec

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    # Parameters
    batch_size   = args.batch_size

    # Get TF cluster and server instances
    cluster, server = TFNode.start_cluster_server(ctx,num_gpus=2,rdma=args.rdma)

    def feed_dict(batch):
        # Convert from [(images, labels)] to two numpy arrays of the proper type
        partitionnum=0
        y = []
        i=0
        for item in batch:
            if(i==0):
              partitionnum=item[0]
              y.append(item[1])
              i=i+1
            else:
              y.append(item[1])
        ys = numpy.array(y)
        ys = ys.astype(numpy.float32)
        xs=numpy.array(range(ys.__len__()))
        xs=xs.astype(numpy.float32)
        return partitionnum,(xs, ys)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        #print("tensorflow model path: {0}".format(logdir))
        tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")

        if(get_available_gpus_len()==0):
            gpu_num="/cpu:0"
        else:
            gpu_num="/gpu:{0}".format(int(ctx.task_index%get_available_gpus_len()))
        print("gpu:=====================",gpu_num)
        logdir=''
        marknum=0
        p_num=0
        # #按gpu个数分发
        with tf.device("/cpu:0"):
            if(args.mode=="train"):
                a_t=tf.get_variable(name='a_t',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer,trainable=False)
                a_t_t_1=tf.get_variable(name='a_t_t_1',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer,trainable=False)
                p_t_t_1=tf.get_variable(name='p_t_t_1',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer,trainable=False)
                p_t_1=tf.get_variable(name='p_t_1',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer,trainable=False)
                #y=tf.get_variable(name='y',dtype=tf.float32,shape=[list_length],initializer=tf.zeros_initializer,trainable=False)
                y=tf.placeholder(dtype=tf.float32, shape=(None))

                T = tf.Variable(1,name="T",dtype=tf.float32) #测量参数
                Z = tf.Variable(1,name="Z",dtype=tf.float32) #测量参数
                H = tf.Variable(0.001,name="H",dtype=tf.float32) #测量系统偏差
                Q = tf.Variable(0.001,name="Q",dtype=tf.float32) #测量系统偏差
                d = tf.Variable(0.05,name="d",dtype=tf.float32)
                c = tf.Variable(0.05,name="c",dtype=tf.float32)
                global_step = tf.Variable(0,dtype=tf.int64,trainable=False)
                for i in range(args.steps):
                    if(tf_feed.should_stop()):
                       tf_feed.terminate()
                    print("--------------------第"+str(ctx.task_index)+"task的第"+str(i+1)+"步迭代---------------------------------")
                    num,(batch_xs, batch_ys) = feed_dict(tf_feed.next_batch(batch_size))
                    if marknum==0 or not str(num).__eq__(p_num):
                       logdir = TFNode.hdfs_path(ctx,str("model/")+args.model+str("_{0}").format(num))
                       marknum=marknum+1
                       p_num=num
                       print("logdir================:",logdir)
                    list_length=batch_ys.__len__()
                    print("batch_ys.__len__():=",batch_ys.__len__())
                    # T = tf.placeholder(dtype=tf.float32, shape=(None)) #状态参数
                    # Q = tf.placeholder(dtype=tf.float32, shape=(None)) #状态偏差
                    # T = tf.placeholder(dtype=tf.float32, shape=(None)) #状态参数
                    # Q = tf.placeholder(dtype=tf.float32, shape=(None)) #状态偏差
                    array_list=[]
                    array_list1=[]
                    var_list=[T,Z,H,Q,d]

                    for j in range(batch_ys.__len__()):
                        if i==0:
                            a_t_t_1=tf.assign(a_t_t_1,T*batch_ys[0]+c)#1
                            p_t_t_1=tf.assign(p_t_t_1,T*Q*T+Q)#2

                            F=Z*p_t_t_1*Z+H#3
                            a_t=tf.assign(a_t,a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys[0]-Z*a_t_t_1-d))#4
                            p_t_1=tf.assign(p_t_1,p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1)#5
                            #预测的y_st
                            array_list.append(Z*a_t+d)
                            array_list1.append(F)
                        else:
                            a_t_t_1=tf.assign(a_t_t_1,T*a_t+c)#1
                            p_t_t_1=tf.assign(p_t_t_1,T*p_t_1*T+Q)#2

                            F=Z*p_t_t_1*Z+H#3
                            a_t=tf.assign(a_t,a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys[i]-Z*a_t_t_1-d))#4
                            p_t_1=tf.assign(p_t_1,p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1)#5
                            #预测的y_st
                            array_list.append(Z*a_t+d)
                            array_list1.append(F)

                    y_st_sum=tf.stack(array_list,axis=-1)
                    F_sum=tf.stack(array_list1,axis=-1)
                    loss=(tf.reduce_sum(tf.log(tf.abs(F_sum)))/2+tf.reduce_sum((y-y_st_sum)*(1/F_sum)*(y-y_st_sum))/2)
                    train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001,rho=0.85).minimize(
                         loss, global_step=global_step)
                    init_op = tf.global_variables_initializer()
                    local_init_op = tf.local_variables_initializer()
                    saver = tf.train.Saver()
                    with tf.Session() as sess:
                        if i==0:
                           sess.run(init_op)
                           sess.run(local_init_op)
                        else:
                           saver.restore(sess,logdir)
                           print("Model restored.")

                        print("before ptimizer:loss=",sess.run(loss,feed_dict={y:batch_ys}))
                        for i in range(int(100)):
                            # sess.run(tf.initialize_all_variables())
                            sess.run(train_op,feed_dict={y:batch_ys})
                        save_path = saver.save(sess, logdir)
                        print("Model saved in file: %s" % save_path)
                        # saver.restore(sess,"/tmp/my-model")
                        print("after ptimizer:loss=",sess.run(loss,feed_dict={y:batch_ys}))
                        print("H:=",sess.run(H))
                    sess.close()
                    # time.sleep((worker_num + 1) * 5)
                tf_feed.terminate()

            else:#测试
                tf.reset_default_graph()
                # Create some variables.
                T = tf.Variable(0.05,name="T",dtype=tf.float32) #测量参数
                Z = tf.Variable(0.05,name="Z",dtype=tf.float32) #测量参数
                H = tf.Variable(0.01,name="H",dtype=tf.float32) #测量系统偏差
                Q = tf.Variable(0.0001,name="Q",dtype=tf.float32) #测量系统偏差
                d = tf.Variable(0.0001,name="d",dtype=tf.float32)
                c = tf.Variable(0.0001,name="c",dtype=tf.float32)
                a_t=tf.get_variable(name='a_t',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer,trainable=False)
                a_t_t_1=tf.get_variable(name='a_t_t_1',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer,trainable=False)
                p_t_t_1=tf.get_variable(name='p_t_t_1',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer,trainable=False)
                p_t_1=tf.get_variable(name='p_t_1',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer,trainable=False)

                # Add ops to save and restore all the variables.
                while not tf_feed.should_stop():
                    num,(batch_xs, batch_ys)= feed_dict(tf_feed.next_batch(batch_size))
                    list_length=batch_ys.__len__()
                    data = {
                        tf.contrib.timeseries.TrainEvalFeatures.TIMES:batch_xs,
                        tf.contrib.timeseries.TrainEvalFeatures.VALUES:batch_ys,
                    }

                    if marknum==0 or not str(num).__eq__(p_num):
                        logdir = TFNode.hdfs_path(ctx,str("model/")+args.model+str("_{0}").format(num))
                        marknum=marknum+1
                        p_num=num

                    init_op = tf.global_variables_initializer()
                    local_init_op = tf.local_variables_initializer()
                    rezult=[]
                    saver = tf.train.Saver()
                    with tf.Session() as sess:
                        saver.restore(sess,logdir)
                        print("Model restored.")
                        # sess.run(init_op)
                        # sess.run(local_init_op)
                        for i in range(batch_ys.__len__()):
                            if i==0:
                                a_t_t_1=tf.assign(a_t_t_1,T*batch_ys[0]+c)#1
                                p_t_t_1=tf.assign(p_t_t_1,T*Q*T+Q)#2

                                F=Z*p_t_t_1*Z+H#3
                                a_t=tf.assign(a_t,a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys[0]-Z*a_t_t_1-d))#4
                                p_t_1=tf.assign(p_t_1,p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1)#5
                                #预测的y_st
                                sess.run([a_t_t_1, p_t_t_1,F,a_t,p_t_1])
                                rep=sess.run(Z*a_t+d)
                                rezult.append((batch_ys[i],rep,batch_ys[i]-rep))
                            else:
                                a_t_t_1=tf.assign(a_t_t_1,T*a_t+c)#1
                                p_t_t_1=tf.assign(p_t_t_1,T*p_t_1*T+Q)#2

                                F=Z*p_t_t_1*Z+H#3
                                a_t=tf.assign(a_t,a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys[i]-Z*a_t_t_1-d))#4
                                p_t_1=tf.assign(p_t_1,p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1)#5
                                #预测的y_st
                                sess.run([a_t_t_1, p_t_t_1,F,a_t,p_t_1])
                                rep=sess.run(Z*a_t+d)
                                rezult.append((batch_ys[i],rep,batch_ys[i]-rep))
                        num_lack=batch_size-result.__len__()
                        if num_lack>0:
                           result.extend([["o","o","o"]]*num_lack)
                        tf_feed.batch_results(result)
            tf_feed.terminate()

conf=SparkConf().setMaster("spark://lf-MS-7976:7077")
sc=SparkContext(conf=conf)
# spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
# sc =spark.sparkContext
# sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

executors = sc._conf.get("spark.executor.instances")
print("executors:=",executors)
num_executors = int(executors) if executors is not None else 4
num_ps = 0

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=10000)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
# parser.add_argument("-f", "--format", help="example format: (csv|pickle|tfr)", choices=["csv","pickle","tfr"], default="csv")
# parser.add_argument("-i", "--images", help="HDFS path to MNIST images in parallelized format")
# parser.add_argument("-l", "--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="ekf_model")
parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=4)
parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=1)
parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
args = parser.parse_args()

#删除存储模型参数用目录
if client_N.list("/user/root/").__contains__("model") and args.mode=='train':
    client_N.delete("/user/root/model/",recursive=True)

print("args:",args)
print("{0} ===== Start".format(datetime.now().isoformat()))

def sample_map(fraction_base,rato):
  def _sample_map(iter):
    while True:
      fraction_use=random.random()
      if fraction_use-rato<0.10 and fraction_use-rato>-0.10:
          break
    input_length=int(fraction_base*fraction_use)
    rezult=[]
    num=0
    # start=random.random()*1000
    for i in iter:
      if num<input_length and 240<num:
        rezult.append(i)
      else:
        if num>input_length:
         break
      num=num+1
    return rezult
  return _sample_map


# dataRDD1_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(1,float(x[1]))).count()
# dataRDD2_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_035FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(2,float(x[1]))).count()
# dataRDD3_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(3,float(x[1]))).count()
# dataRDD4_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_041FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(4,float(x[1]))).count()
# dataRDD5_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(5,float(x[1]))).count()
# dataRDD6_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_035FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(6,float(x[1]))).count()
# dataRDD7_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(7,float(x[1]))).count()
# dataRDD8_count=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_041FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(8,float(x[1]))).count()

fraction_base,rato=5000,0.75
dataRDD1=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt")\
.map(lambda x:str(x).split(",")).map(lambda x:("34FS",float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
dataRDD2=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_035FS001.txt") \
    .map(lambda x:str(x).split(",")).map(lambda x:("35FS",float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
dataRDD3=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt") \
    .map(lambda x:str(x).split(",")).map(lambda x:("39FS",float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
dataRDD4=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_041FS001.txt") \
    .map(lambda x:str(x).split(",")).map(lambda x:("41FS",float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
# dataRDD5=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(5,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
# dataRDD6=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_035FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(6,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
# dataRDD7=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(7,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)
# dataRDD8=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_041FS001.txt") \
#     .map(lambda x:str(x).split(",")).map(lambda x:(8,float(x[1]))).mapPartitions(sample_map(fraction_base,rato)).repartition(1)

a=[dataRDD1,dataRDD2,dataRDD3,dataRDD4]
#,dataRDD5,dataRDD6,dataRDD7,dataRDD8]
dataRDD=sc.union(a)
print(dataRDD.take(100))
rdd_count=dataRDD.count()
print("count:====================",rdd_count)
print("partition:=",dataRDD.getNumPartitions())
# def func(x,iter):
#     result = []
#     for value in iter:
#         result.append((x,value))
#     return result
# dataRDD=sc.parallelize(range(150003),3).mapPartitionsWithIndex(func)
# print(dataRDD.take(100))

if rdd_count<1000:
   args.epochs=2
   args.batch_size=1000
else:
   args.epochs=1
   args.batch_size=1000

# print("getNumPartitions:=",dataRDD.getNumPartitions())
cluster = TFCluster.run(sc, map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
# if args.mode == "train":
cluster.train(dataRDD, args.epochs)
cluster.shutdown()
print("-----------------train over-------------------------------")
# # else:
def func1(iter):
    result = []
    num=0
    for value in iter:
        result.append(value)
        num=num+1
        if num>240:
            break
    return result
dataRDD1=sc.union(a).mapPartitions(func1)
# dataRDD1=sc.parallelize(range(600),3).mapPartitionsWithIndex(func)
# print("getNumPartitions:=",dataRDD1.getNumPartitions())
args.mode='inference'
args.batch_size=240
args.epochs=1
print(args.mode)
cluster1 = TFCluster.run(sc, map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
labelRDD = cluster1.inference(dataRDD1)
print(labelRDD.filter(lambda x:not str(x[0]).__eq__('o')).collect())# .saveAsTextFile(args.output)
cluster1.shutdown()
print("-----------------inference over-------------------------------")
print("{0} ===== Stop".format(datetime.now().isoformat()))