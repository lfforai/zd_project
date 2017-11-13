#///////////////////////////////////////////////////////////////////////////////////////

#./spark-submit --conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" --conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4"  ~/IdeaProjects/pyspark_t/.idea/spark-warehouse/spark-streaming/exp.py


from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import tensorflow as tf

def print_log(worker_num, arg):
    print("{0}: {1}".format(worker_num, arg))

def get_available_gpus_len():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU'].__len__()

def normal_probability(min,max,y,n=50000,p=0.95,gpu_num="0"):
    #用反复逼近的方式迭代求最接近最大值的
    import numpy as np
    import tensorflow as tf

    def normal_probability_density(y_input,h=-1):
        #均值=0,标准差=h的正态分布的概率密度函数
        def _map_func(x):
            pi=3.141592654
            if h==-1:#1/n^(0.2)
                h=tfy.shape[0]
                result=tf.reduce_mean(tf.exp(-tf.pow(tf.exp(x-y_input),2)/(tf.pow(h)*2.0))/(tf.pow(pi*2,0.5)*h))
            else:
                result=tf.reduce_mean(tf.exp(-tf.pow(tf.exp(x-y_input),2)/(tf.pow(h)*2.0))/(tf.pow(pi*2,0.5)*h))
            return rezult
        return _map_func

    with tf.device("/cpu:"+str(gpu_num)):
         #y_ts=tf.placeholder(dtype=tf.float32, shape=(None))
         y_ts=tf.convert_to_tensor(y)
         value_big=tf.get_variable("value_big",shape=[],dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer)
         value_little=tf.get_variable("value_little",shape=[],dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer)
         value_now=tf.get_variable("value_now",shape=[],dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer)
         min_cast=tf.constant(name="min_cast",value=min)#下限
         config = tf.ConfigProto()#luofeng jia
         config.gpu_options.allow_growth=True
         #赋值
         value_big=tf.assign(value_big,max)
         value_little=tf.assign(value_little,min)
         value_now=tf.assign(value_now,value_big)
         dx=(value_now-value_little)/n
         p_now_np=0
         init_op = tf.global_variables_initializer()
         local_init_op = tf.local_variables_initializer()
         with tf.Session(config=config) as sess:
              while True:
                    p_now=tf.reduce_sum(tf.data.Dataset.from_tensor_slices(min_cast+tf.convert_to_tensor(linspace(1,n,n))*dx) \
                                        .map(normal_probability_density(y))
                                        .map(lambda x:x*dx)
                                        )
                    p_now_np=sess.run(p_now,feed_dict={y:batch_ys})
                    if p_now_np-p<0.005 and p_now_np-p>-0.005:
                        rezult=sess.run(value_now,feed_dict={y:batch_ys})
                        break
                    else:
                        if p_now_np>p:#big保留,little调整
                            value_big=value_now
                            value_now=(value_big-value_little)/2
                        else:
                            value_little=value_now
                            value_now=(value_big-value_little)/2
    return p_now_np,rezult

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
        with tf.device(gpu_num):
            if(args.mode=="train"):
              print("no need train")
            else:#测试
              # Add ops to save and restore all the variables.
              num,(batch_xs, batch_ys) = feed_dict(tf_feed.next_batch(batch_size))
              y=tf.placeholder(dtype=tf.float32, shape=(None))
              #寻找F（x）大于95%或者5%的异常值点
              max_limit=tf.reduce_min(y)+1000
              min_limit=tf.reduce_max(y)+1000
              p_95=normal_probability(min_limit,max_limit,p=0.95)
              p_05=normal_probability(min_limit,max_limit,p=0.05)








