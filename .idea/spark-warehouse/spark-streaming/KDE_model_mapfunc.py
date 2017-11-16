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

#用反复逼近求解法计算分位数
def normal_probability(y,n=10000,p=0.95,gpu_num="0"):
    #用反复逼近的方式迭代求最接近最大值的
    import numpy as np
    import tensorflow as tf

    # with tf.device("/gpu:"+str(gpu_num)):
    #y_in=tf.placeholder(dtype=tf.float32,shape=(None))
    print(y)

    def normal_probability_density(y_in,x_t):
        #均值=0,标准差=h的正态分布的概率密度函数
        def map_func(x_s):
            pi=tf.constant(3.141592654,dtype=tf.float32)
            h=1.0/tf.pow(tf.constant(y.__len__(),dtype=tf.float32),0.2)
            result_in=tf.reduce_mean(tf.exp(-tf.pow(x_s-x_t-y_in,2)/(tf.pow(h,2)*2.0))/(tf.pow(pi*2.0,0.5)*h))
            return result_in
        return map_func

    # def normal_probability_density(x_s,y_in,x_t):
    #     #均值=0,标准差=h的正态分布的概率密度函数
    #     pi=tf.constant(3.141592654,dtype=tf.float32)
    #     h=1.0/tf.pow(tf.constant(y.__len__(),dtype=tf.float32),0.2)
    #     result_in=tf.reduce_mean(tf.exp(-tf.pow((x_s-x_t-y_in),2)/(tf.pow(h,2)*2.0))/(tf.pow(pi*2.0,0.5)*h))
    #     # print(tf.Session().run(result_in))
    #     return result_in

    config = tf.ConfigProto()#luofeng jia
    config.gpu_options.allow_growth=True

    value_big=tf.get_variable("value_big",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer)
    value_little=tf.get_variable("value_little",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer)
    value_now=tf.get_variable("value_now",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer)
    result=tf.get_variable("result",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer)
    dx=tf.get_variable("dx",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer)
    x1=tf.get_variable("x1",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer)
    iterator=tf.get_variable("iterator",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer)

    #赋值
    y_in=tf.convert_to_tensor(y,dtype=tf.float32)
    min_cast=tf.convert_to_tensor(y.min()-2,dtype=tf.float32)#下限
    list_dx=tf.convert_to_tensor(np.linspace(1,n,n),dtype=tf.float32)

    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()

    value_big_tmp=y.max()+2
    value_little_tmp=y.min()-2
    value_now_tmp=value_big_tmp

    result=0
    while True:
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            sess.run(local_init_op)
            value_big=tf.convert_to_tensor(value_big_tmp,dtype=tf.float32)
            value_little=tf.convert_to_tensor(value_little_tmp,dtype=tf.float32)
            value_now=tf.convert_to_tensor(value_now,dtype=tf.float32)
            result=0
            p_now_np=0
            dx=(value_now-min_cast)/n
            x1=min_cast+list_dx*dx
            print(sess.run(x1))

            rdd_dataset=tf.contrib.data.Dataset.from_tensor_slices(x1).map(normal_probability_density(y_in,dx/2),num_threads=2).map(lambda x2:x2*dx,num_threads=2)

            iterator = rdd_dataset.make_initializable_iterator()
            sess.run(iterator.initializer)

            for i in range(int(n)):
                try:
                    result=result+iterator.get_next()
                except tf.errors.OutOfRangeError:
                    break
                    # for i  in range(n):
                    #   result=result+(normal_probability_density(x1[i],dx/2,y_in))*dx

                    # print("result:=",
            p_now_np=sess.run(result)
            print("p_now_np",p_now_np)
            if np.abs(p_now_np-p)<0.001:
                result=sess.run(value_now)
                break
            else:
                print("继续")
                if p_now_np>p:#big保留,little调整
                    value_big=value_now
                    value_now=(value_big+value_little)/2
                    value_now_tmp=sess.run(value_now)
                    value_big_tmp=sess.run(value_big)
                    value_little_tmp=sess.run(value_little)
                else:
                    value_little=value_now
                    value_now=(value_big+value_little)/2
                    value_now_tmp=sess.run(value_now)
                    value_big_tmp=sess.run(value_big)
                    value_little_tmp=sess.run(value_little)
        sess.close()
    return p_now_np,result

def map_func(args, ctx):
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
        x = []
        i=0
        for item in batch:
            if(i==0):
                partitionnum=item[0]
                y.append(item[3])
                x.append([item[1],item[2],item[4]])
                i=i+1
            else:
                y.append(item[3])
                x.append([item[1],item[2],item[4]])
        ys = numpy.array(y)
        ys = ys.astype(numpy.float32)
        return partitionnum,(x, ys)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        #print("tensorflow model path: {0}".format(logdir))
        tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")

        if(get_available_gpus_len()==15):
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
                 print("no need train")
            else:#测试
                while not tf_feed.should_stop():
                    # Add ops to save and restore all the variables.
                    num,(batch_xs, batch_ys) = feed_dict(tf_feed.next_batch(batch_size))

                    #寻找F（x）大于95%或者5%的异常值点
                    p_up,p_value_up=normal_probability(batch_ys,n=3000,p=0.95,gpu_num="0")
                    print("上异常点概率：=%f，分位值：=%f"%(p_up,p_value_up))

                    p_down,p_value_down=normal_probability(batch_ys,n=3000,p=0.05,gpu_num="0")
                    print("下异常点概率：=%f，分位值：=%f"%(p_down,p_value_down))

                    rezult=filter(lambda x:x[0]>p_value_up or x[0]<p_value_down,list[zip(batch_ys,batch_xs)])
                    num_lack=batch_size-results.__len__()
                    if num_lack>0:
                        results.extend([["o","o"]]*num_lack)
                    tf_feed.batch_results(results)