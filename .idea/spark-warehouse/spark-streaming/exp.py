#///////////////////////////////////////////////////////////////////////////////////////

#./spark-submit --conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" --conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4"  ~/IdeaProjects/pyspark_t/.idea/spark-warehouse/spark-streaming/exp.py



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
        y = []
        for item in batch:
            y.append(item)
        ys = numpy.array(y)
        ys = ys.astype(numpy.float32)
        xs=numpy.array(range(ys.__len__()))
        xs=xs.astype(numpy.float32)
        return (xs, ys)
        global client


    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        logdir = TFNode.hdfs_path(ctx,str("model/")+args.model+str("_{0}").format(ctx.task_index))
        print("tensorflow model path: {0}".format(logdir))
        tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")

        if(get_available_gpus_len()==0):
            gpu_num="/cpu:0"
        else:
            gpu_num="/gpu:{0}".format(int(ctx.task_index%get_available_gpus_len()))
        print("gpu:=====================",gpu_num)
        ar = tf.contrib.timeseries.ARRegressor(
            periodicities=200, input_window_size=30, output_window_size=10,
            num_features=1,
            loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS,model_dir=logdir)

        # #按gpu个数分发
        with tf.device(gpu_num):
            if(args.mode=="train"):
                for i in range(args.steps):
                    if(tf_feed.should_stop()):
                       tf_feed.terminate()
                    print("--------------------"+str(ctx.task_index)+":"+str(i)+"---------------------------------")
                    batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
                    data = {
                        tf.contrib.timeseries.TrainEvalFeatures.TIMES:batch_xs,
                        tf.contrib.timeseries.TrainEvalFeatures.VALUES:batch_ys,
                    }
                    reader = NumpyReader(data)
                    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=16, window_size=40)
                    ar.train(input_fn=train_input_fn, steps=500)
                tf_feed.terminate()

            else:#测试
              while not tf_feed.should_stop():
                batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
                data = {
                    tf.contrib.timeseries.TrainEvalFeatures.TIMES:batch_xs,
                    tf.contrib.timeseries.TrainEvalFeatures.VALUES:batch_ys,
                }
                reader_N = NumpyReader(data)
                evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader_N)
                #keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
                evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)
                _y=list(evaluation['mean'].reshape(-1))
                y=list(data['values'].reshape(-1))
                results =[[e,l,e-l] for e,l in zip(_y,y)]
                num_lack=batch_size-results.__len__()
                if num_lack>0:
                   results.extend([["o","o","o"]]*num_lack)
                tf_feed.batch_results(results)
              tf_feed.terminate()

conf=SparkConf().setMaster("spark://lf-MS-7976:7077")
sc=SparkContext(conf=conf)
# spark = sql_n.SparkSession.builder.appName("lf").config(conf=conf).getOrCreate()
# sc =spark.sparkContext
# sqlContext=sql_n.SQLContext(sparkContext=sc,sparkSession=spark)

executors = sc._conf.get("spark.executor.instances")
print("executors:=",executors)
num_executors = int(executors) if executors is not None else 3
num_ps = 0

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=50000)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
# parser.add_argument("-f", "--format", help="example format: (csv|pickle|tfr)", choices=["csv","pickle","tfr"], default="csv")
# parser.add_argument("-i", "--images", help="HDFS path to MNIST images in parallelized format")
# parser.add_argument("-l", "--labels", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="AR_model")
parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=1)
parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=10)
parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("-X", "--mode", help="train|inference", default="train")
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False)
args = parser.parse_args()

#删除存储模型参数用目录
if client_N.list("/user/root/").__contains__("model") and args.mode=='train':
    client_N.delete("/user/root/model/",recursive=True)

print("args:",args)
print("{0} ===== Start".format(datetime.now().isoformat()))

dataRDD1=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt",1)\
.map(lambda x:str(x).split(",")).map(lambda x:float(x[1])).repartition(1).persist()
dataRDD2=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt",1) \
    .map(lambda x:str(x).split(",")).map(lambda x:float(x[1])).repartition(1).persist()
# dataRDD3=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_039FS001.txt",1) \
#     .map(lambda x:str(x).split(",")).map(lambda x:float(x[1])).repartition(1)
dataRDD4=sc.textFile("hdfs://127.0.0.1:9000/zd_data2/FS/G_CFYH_2_034FS001.txt",1) \
    .map(lambda x:str(x).split(",")).map(lambda x:float(x[1])).repartition(1).persist()
a=[dataRDD1,dataRDD2,dataRDD4]
dataRDD=sc.union(a)
# print(dataRDD.take(100))
print("partition:=",dataRDD.getNumPartitions())
# dataRDD=sc.parallelize(range(100000),4)
# print("getNumPartitions:=",dataRDD.getNumPartitions())
cluster = TFCluster.run(sc, map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
# if args.mode == "train":
cluster.train(dataRDD, args.epochs)
cluster.shutdown()
print("-----------------train over-------------------------------")
# # else:
dataRDD=sc.union(a)
args.mode='inference'
args.batch_size=40000
print(args.mode)
cluster = TFCluster.run(sc, map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
labelRDD = cluster.inference(dataRDD)
print(labelRDD.filter(lambda x:not str(x[0]).__eq__('o')).take(1000))# .saveAsTextFile(args.output)
cluster1.shutdown()
print("-----------------inference over-------------------------------")
print("{0} ===== Stop".format(datetime.now().isoformat()))