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

def map_func_cluster(args, ctx):
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
        list_xs_1=[]
        partitionnum=0
        y = []
        i=0
        for item in batch:
            # print(item)
            y.append([item[1],item[2]])
            list_xs_1.append(item[0])
        ys = numpy.array(y)
        ys = ys.astype(numpy.float64)
        return list_xs_1,ys #ys={平均值,标准差}

        ############kmeans算法计算###############
    def feed_dict_fence2(batch):
        # Convert from [(images, labels)] to two numpy arrays of the proper type
        list_xs_1=[]
        partitionnum=0
        y = []
        i=0
        for item in batch:
            # print(item)
            y.append(item[1])
            list_xs_1.append(item[0])
        ys = numpy.array(y)
        ys = ys.astype(numpy.float64)
        return list_xs_1,ys #ys={平均值,标准差}

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
                tf_feed.terminate()
            else:#测试"inference"
                # Add ops to save and restore all the variables
                i_n=0
                k=2
                limit=0.50
                list_length_first=0
                while not tf_feed.should_stop():
                    results=[]#有问题的选项直接输出
                    print("------------------第"+str(i_n+1)+"次 ekf—batch inference-----------------------")
                    xs_info,batch_ys = feed_dict_fence(tf_feed.next_batch(batch_size))
                    for jj in range(batch_ys.__len__()):
                        print(xs_info[jj])
                        print(batch_ys[jj])
                        print("------------")
                    # print(xs_info,xs_info[0])
                    if (i_n==0):
                       list_length=batch_ys.__len__()
                       list_length_first=list_length
                    else:
                       list_length=batch_ys.__len__()

                    print("length:=========",list_length)

                    if list_length>0:
                        step=0
                        while(step<args.steps):
                            #在聚类前先剔除距离中心距离３倍标准差以外的异常距离点
                            # print(batch_ys)
                            center_befor=np.mean(batch_ys)
                            # print("中心点：＝==", center_befor)
                            #计算到中心点距离
                            Radius_befor=np.sqrt(np.sum(np.power(batch_ys-center_befor,2),axis=1))
                            from numpy import std
                            std_value=std(Radius_befor)
                            avg_value=numpy.average(Radius_befor)
                            out_value=avg_value+3*std_value#超出部分视为异常值不参与聚类
                            # print("3倍方差以外距离：＝", out_value)
                            org=zip(xs_info,batch_ys)
                            together=zip(Radius_befor,org)


                            #保留三倍标标准差之内的点
                            batch_ys_info=[[e[1][0],e[1][1]] for e in\
                                           list(filter(lambda x:float(x[0])<out_value,together))]
                            xs_info=list(map(lambda x:x[0],batch_ys_info))
                            batch_ys=numpy.array(list(map(lambda x:x[1],batch_ys_info)))


                            #剔除三倍标标准差之外的点
                            batch_ys_info=[[e[1][0],e[1][1]] for e in \
                                           list(filter(lambda x:float(x[0])>=out_value,together))]
                            xs_info_out=list(map(lambda x:x[0],batch_ys_info))
                            #batch_ys_out=np.asarray(list(map(lambda x:float(x[1]),batch_ys_info)))
                            # for jj in range(batch_ys.__len__()):
                            #     print(xs_info[jj])
                            #     print(batch_ys[jj])
                            #     print("------------")
                            center,result=KMeansCluster(batch_ys,k);
                            # print("已经完成聚类计算")
                            # print("center:=",center)
                            #计算样本到每个类中心的半径距离
                            list_centers=[]
                            for i in result:
                                list_centers.append(center[i])

                            Radius=np.sqrt(np.sum(np.power(batch_ys-list_centers,2),axis=1))

                            ave_Radius=[[i,0,0,0] for i in range(k)]#计算平均半径[类类名字,半径平均值,类中样本个数,标准差]

                            # 计算均值和样本个数
                            for i in range(result.__len__()):
                                ave_Radius[result[i]][1]=ave_Radius[result[i]][1]+Radius[i]
                                ave_Radius[result[i]][2]=ave_Radius[result[i]][2]+1

                            ave_Radius=np.array(list(map(lambda x:[x[0],x[1]/x[2],x[2],x[3]],ave_Radius)))

                            #计算标准差
                            for i in range(result.__len__()):
                                ave_Radius[result[i]][3]=ave_Radius[result[i]][3]+np.power(ave_Radius[result[i]][1]-Radius[i],2)

                            for i in range(k):
                                if ave_Radius[i][2]!=1:
                                    ave_Radius[i][3]=np.sqrt(ave_Radius[i][3]/(ave_Radius[i][2]-1))
                                else:
                                    ave_Radius[i][3]=0

                            ave_Radius_out=[[i[1]+3*i[3]]  for i in ave_Radius]
                            list_rad_out=[]
                            for i in result:
                                list_rad_out.append(ave_Radius_out[i])

                            last=[[e,l[0],l[1][0]] for e,l in zip(xs_info,zip(Radius, list_rad_out))]
                            last_n=list(map(lambda x:x[0],filter(lambda x:True if x[1]==1 else False,map(lambda x:[x[0],1] if x[1]>x[2] else [x[0],0],last))))

                            last_n.extend(xs_info_out)#把异常数据的添加进来,直接作为三倍标准差以外的点
                            last_set=set(last_n)

                            rezult_list=[]
                            for item in last_set:
                                rezult_list.append([item,last_n.count(item)])

                            a_n=[]
                            for i in rezult_list:
                                a_n.append(float(i[1]))

                            total=sum(a_n)

                            last_one=[[item[0],float(item[1]/total)] for item in rezult_list]
                            max_value=0

                            rezult=''
                            for i in last_one:
                                if max_value<i[1]:
                                   max_value=i[1]
                                   rezult=str(i[0])
                            print("max:===",max_value)
                            print("max:rezult===:",rezult)
                            if(max_value>limit):
                              results.append(rezult)
                              temp_list_1=[[e,l[0],l[1]] for e,l in zip(xs_info,batch_ys)]
                              temp_list_2=list(filter(lambda x:not str(x[0]).__eq__(rezult),temp_list_1))
                              # for j_j in range(10):
                              #     print(temp_list_2[j_j])
                              xs_info_out=list(filter(lambda x:not str(x[0]).__eq__(rezult),xs_info_out))
                              xs_info=[str(temp_value[0]) for temp_value in temp_list_2]
                              batch_ys=numpy.array([[float(temp_value[1]),float(temp_value[2])] for temp_value in temp_list_2])
                              list_length=batch_ys.__len__()
                              # print("list_length:====",list_length)
                              if list_length<k:
                                  step=1000
                                  break
                              else:
                                  pass
                                  # for j in range(batch_ys.__len__()):
                                  #   print("2=first:==",batch_ys[j])
                                  #   print("2=secend:==",xs_info[j])
                            step=step+1
                            #如果超过半径的类中一类样本的点超过５０％，可以怀疑为异常数据
                            # statis=[[i,dict()] for i in range(K)]#每一组的统计信息记录
                            #
                            # result =[[e,l] for e,l in zip(value,xs_info)] #e:=分的类别,l:=具体的值
                            #
                            #统计每个源点在每个分组中出现的次数
                            # for i in result:
                            #    if statis[i[0]][1].keys().__contains__(str(i[1])):#已经有了
                            #       statis[i[0]][1][str(i[1])]=int(statis[i[0]][1][str(i[1])])+1
                            #    else:
                            #        statis[i[0]][1][str(i[1])]=1

                            #大于对于每个组中大于占比５０％的分组怀疑是有问题的点，提取出来作为异常点
                            # for i in range(K):
                            #     #每一组
                            #    total_num=0
                            #
                            #    for key in statis[i][1].keys():
                            #        total_num=total_num+statis[i][1][str(key)]#组里面的样本总数
                            #
                            #    max_w=0
                            #    load_key=''
                            #    for key in statis[i][1].keys():#占比最大的样本类
                            #        if max_w==0:
                            #           max_w=float(statis[i][1][str(key)]/total_num)
                            #           load_key=str(key)
                            #        else:
                            #           if max_w<float(statis[i][1][str(key)]/total_num):
                            #              max_w=float(statis[i][1][str(key)]/total_num)
                            #              load_key=str(key)
                            #           else:
                            #              pass
                            #
                            #    print("max_w:={%f},key={%s}"%(max_w,load_key))
                            #
                            #    if max_w>limit:
                            #       result_N =[[e,l] for e,l in zip(batch_ys,xs_info)]
                            #       print("max_w:={%f},key={%s}"%(max_w,load_key))
                            #       results=filter(lambda x:str(x[1]).__eq__(load_key),result_N)
                            #    else:
                            #       pass
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