from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import numpy as np

def print_log(worker_num, arg):
    print("{0}: {1}".format(worker_num, arg))

def get_available_gpus_len():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU'].__len__()

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

    # -*- coding: utf-8 -*-
    ############Sachin Joglekar的基于tensorflow写的一个kmeans模板###############
    def KMeansCluster(vectors, noofclusters):
        import numpy as np
        from numpy.linalg import cholesky
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import tensorflow as tf
        from random import choice, shuffle
        from numpy import array

        """
        K-Means Clustering using TensorFlow.
        `vertors`应该是一个n*k的二维的NumPy的数组，其中n代表着K维向量的数目
        'noofclusters' 代表了待分的集群的数目，是一个整型值
        """

        """
        K-Means Clustering using TensorFlow.
        `vertors`应该是一个n*k的二维的NumPy的数组，其中n代表着K维向量的数目
        'noofclusters' 代表了待分的集群的数目，是一个整型值
        """

        noofclusters = int(noofclusters)
        assert noofclusters < len(vectors)
        #找出每个向量的维度
        dim = len(vectors[0])
        #辅助随机地从可得的向量中选取中心点
        vector_indices = list(range(len(vectors)))
        shuffle(vector_indices)
        #计算图
        #我们创建了一个默认的计算流的图用于整个算法中，这样就保证了当函数被多次调用      #时，默认的图并不会被从上一次调用时留下的未使用的OPS或者Variables挤满
        graph = tf.Graph()
        with graph.as_default():
            #计算的会话
            sess = tf.Session()
            ##构建基本的计算的元素
            ##首先我们需要保证每个中心点都会存在一个Variable矩阵
            ##从现有的点集合中抽取出一部分作为默认的中心点
            centroids = [tf.Variable((vectors[vector_indices[i]]))
                         for i in range(noofclusters)]
            ##创建一个placeholder用于存放各个中心点可能的分类的情况
            centroid_value = tf.placeholder("float64", [dim])
            cent_assigns = []
            for centroid in centroids:
                cent_assigns.append(tf.assign(centroid, centroid_value))
            ##对于每个独立向量的分属的类别设置为默认值0
            assignments = [tf.Variable(0) for i in range(len(vectors))]
            ##这些节点在后续的操作中会被分配到合适的值
            assignment_value = tf.placeholder("int32")
            cluster_assigns = []
            for assignment in assignments:
                cluster_assigns.append(tf.assign(assignment,
                                                 assignment_value))
            ##下面创建用于计算平均值的操作节点
            #输入的placeholder
            mean_input = tf.placeholder("float", [None, dim])
            #节点/OP接受输入，并且计算0维度的平均值，譬如输入的向量列表
            mean_op = tf.reduce_mean(mean_input, 0)
            ##用于计算欧几里得距离的节点
            v1 = tf.placeholder("float", [dim])
            v2 = tf.placeholder("float", [dim])
            euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))
            ##这个OP会决定应该将向量归属到哪个节点
            ##基于向量到中心点的欧几里得距离
            #Placeholder for input
            centroid_distances = tf.placeholder("float", [noofclusters])
            cluster_assignment = tf.argmin(centroid_distances, 0)
            ##初始化所有的状态值
            ##这会帮助初始化图中定义的所有Variables。Variable-initializer应该定
            ##义在所有的Variables被构造之后，这样所有的Variables才会被纳入初始化
            init_op = tf.global_variables_initializer()
            #初始化所有的变量
            sess.run(init_op)
            ##集群遍历
            #接下来在K-Means聚类迭代中使用最大期望算法。为了简单起见，只让它执行固
            #定的次数，而不设置一个终止条件
            noofiterations = 20
            for iteration_n in range(noofiterations):

                ##期望步骤
                ##基于上次迭代后算出的中心点的未知
                ##the _expected_ centroid assignments.
                #首先遍历所有的向量
                for vector_n in range(len(vectors)):
                    vect = vectors[vector_n]
                    #计算给定向量与分配的中心节点之间的欧几里得距离
                    distances = [sess.run(euclid_dist, feed_dict={
                        v1: vect, v2: sess.run(centroid)})
                                 for centroid in centroids]
                    #下面可以使用集群分配操作，将上述的距离当做输入
                    assignment = sess.run(cluster_assignment, feed_dict = {
                        centroid_distances: distances})
                    #接下来为每个向量分配合适的值
                    sess.run(cluster_assigns[vector_n], feed_dict={
                        assignment_value: assignment})

                ##最大化的步骤
                #基于上述的期望步骤，计算每个新的中心点的距离从而使集群内的平方和最小
                for cluster_n in range(noofclusters):
                    #收集所有分配给该集群的向量
                    # assigned_vects = [vectors[i] for i in range(len(vectors))
                    #                   if sess.run(assignments[i]) == cluster_n]
                    assigned_vects=[]
                    for i in range(len(vectors)):
                        if sess.run(assignments[i]) == cluster_n:
                            assigned_vects.append(vectors[i])

                    if assigned_vects.__len__()!=0:
                        #计算新的集群中心点
                        new_location = sess.run(mean_op, feed_dict={
                            mean_input: array(assigned_vects)})
                        #为每个向量分配合适的中心点
                        sess.run(cent_assigns[cluster_n], feed_dict={
                            centroid_value: new_location})

            #返回中心节点和分组
            centroids = sess.run(centroids)
            assignments = sess.run(assignments)
            return centroids, assignments
    ############生成测试数据###############


    ############kmeans算法计算###############
    def feed_dict_fence(batch):
        # Convert from [(images, labels)] to two numpy arrays of the proper type
        partitionnum=0
        ys = []
        i=0
        each_y_value=[]#每个组里面一个是一个源点的样本
        info=""
        for item in batch:
            if i==0:
               info=item[0]#源点名字
               each_y_value.append(item[1])#源点数据
               i=i+1
            else:
               if(str(info).__eq__(item[0])):
                  each_y_value.append(item[1])
               else:
                  ys.append([info,each_y_value])
                  each_y_value=[]
                  info=item[0]#源点名
                  each_y_value.append(item[1])#源点数据
        return ys #[原点名字，list[数据]]

    if job_name == "ps":
       server.join()
    elif job_name == "worker":
        #print("tensorflow model path: {0}".format(logdir))
        tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train")
        # #按gpu个数分发
        with tf.device("/cpu:0"):
            if(args.mode=="train"):
                tf_feed.terminate()
            else:#测试"inference"
                i_n=0
                while not tf_feed.should_stop():
                    results=[]#有问题的选项直接输出
                    print("------------------第"+str(i_n+1)+"次 spearman—batch inference-----------------------")
                    batch_ys = feed_dict_fence(tf_feed.next_batch(batch_size))
                    list_length_first=batch_ys.__len__()

                    if  list_length_first>0:
                        info_N=np.zeros([list_length_first,list_length_first])

                        pearson_out_module = tf.load_op_library('/tensorflow_user_lib/pearson_out.so')
                        temp_shape=tf.zeros([1])#传递shape的参数

                        with tf.Session() as sess:
                            for i in range(list_length_first):
                                for j in range(list_length_first):
                                    info_N[i][j]= sess.run(pearson_out_module.pearson_out(batch_ys[i][1],batch_ys[j][1],sess.run(temp_shape)))
                                    print("one:=%s,two=%s,r=%f"%(batch_ys[i][0],batch_ys[i][1],info_N[i][j]))

                        num_lack=list_length_first-results.__len__()
                        if num_lack>0:
                            results.extend([["o","o","o"]]*num_lack)
                        tf_feed.batch_results(results)
                    else:
                        print("最后一个了！")
                        results=[]
                        tf_feed.batch_results(results)
                    i_n=i_n+1
            tf_feed.terminate()