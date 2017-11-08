from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

def print_log(worker_num, arg):
    print("{0}: {1}".format(worker_num, arg))

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
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    # Parameters
    batch_size   = args.batch_size

    # Get TF cluster and server instances
    cluster, server = TFNode.start_cluster_server(ctx,1,rdma=args.rdma)

    def feed_dict(batch):
        # Convert from [(images, labels)] to two numpy arrays of the proper type
        y = []
        for item in batch:
            y.append(item[0])
        ys = numpy.array(y)
        ys = xs.astype(numpy.float32)
        xs=numpy.array(range(xs.__len__()))
        xs=xs.astype(numpy.float32)
        return (xs, ys)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        logdir = TFNode.hdfs_path(ctx, args.model)
        tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")
        batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
        feed = {x: batch_xs, y_: batch_ys}

        data = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
        }

        reader = NumpyReader(data)

        train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
            reader, batch_size=16, window_size=40)

        ar = tf.contrib.timeseries.ARRegressor(
            periodicities=200, input_window_size=30, output_window_size=10,
            num_features=1,
            loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS,model_dir="hdfs:")

        ar.train(input_fn=train_input_fn, steps=args.seteps)

        # evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
        # # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
        # evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

