#./spark-submit --py-files ~/IdeaProjects/zd_project/.idea/spark-warehouse/spark-streaming/sample_model.py --py-files ~/IdeaProjects/zd_project/.idea/spark-warehouse/spark-streaming/AR_model_mapfunc.py  --conf spark.executorEnv.LD_LIBRARY_PATH="${JAVA_HOME}/jre/lib/amd64/server:/usr/local/cuda-8.0/lib64"  --conf spark.executorEnv.CLASSPATH="$($HADOOP_HOME/bin/hadoop classpath --glob):${CLASSPATH}" --conf spark.executorEnv.HADOOP_HDFS_HOME="/tool_lf/hadoop/hadoop-2.7.4"  ~/IdeaProjects/zd_project/.idea/spark-warehouse/spark-streaming/model_run.py
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

def print_log(worker_num, arg):
    print("{0}: {1}".format(worker_num, arg))
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
                print(item)
                partitionnum=item[0]
                y.append(item[1][1])
                i=i+1
            else:
                y.append(item[1][1])
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
                i=0
                while not tf_feed.should_stop() and i<5:
                    # if(tf_feed.should_stop()):
                    #     tf_feed.terminate()
                    print("--------------------第"+str(ctx.task_index)+"task的第"+str(i+1)+"步迭代---------------------------------")
                    num,(batch_xs, batch_ys) = feed_dict(tf_feed.next_batch(batch_size))
                    data = {
                        tf.contrib.timeseries.TrainEvalFeatures.TIMES:batch_xs,
                        tf.contrib.timeseries.TrainEvalFeatures.VALUES:batch_ys,
                    }

                    if marknum==0 or num!=p_num:
                        logdir = TFNode.hdfs_path(ctx,str("model/")+args.model+str("_{0}").format(num))
                        marknum=marknum+1
                        p_num=num
                        print("logdir================:",logdir)
                    ar = tf.contrib.timeseries.ARRegressor(
                        periodicities=200, input_window_size=30, output_window_size=10,
                        num_features=1,
                        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS,model_dir=logdir)
                    reader = NumpyReader(data)
                    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=200, window_size=40)
                    ar.train(input_fn=train_input_fn, steps=100)
                    i=i+1
                    # time.sleep((worker_num + 1) * 5)
                tf_feed.terminate()

            else:#测试
                while not tf_feed.should_stop():
                    num,(batch_xs, batch_ys)= feed_dict(tf_feed.next_batch(batch_size))
                    data = {
                        tf.contrib.timeseries.TrainEvalFeatures.TIMES:batch_xs,
                        tf.contrib.timeseries.TrainEvalFeatures.VALUES:batch_ys,
                    }
                    if marknum==0 or num!=p_num:
                        logdir = TFNode.hdfs_path(ctx,str("model/")+args.model+str("_{0}").format(num))
                        marknum=marknum+1
                        p_num=num
                    ar = tf.contrib.timeseries.ARRegressor(
                        periodicities=200, input_window_size=30, output_window_size=10,
                        num_features=1,
                        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS,model_dir=logdir)
                    reader_N = NumpyReader(data)
                    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader_N)
                    #keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
                    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)
                    _y=list(evaluation['mean'].reshape(-1))
                    y=list(data['values'].reshape(-1))
                    results =[[e,l,e-l] for e,l in zip(_y,y)]
                    # results =[(ctx.task_index,e) for e in batch_ys]
                    num_lack=batch_size-results.__len__()
                    if num_lack>0:
                        results.extend([["o","o","o"]]*num_lack)
                    tf_feed.batch_results(results)
            tf_feed.terminate()