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
        ys = []
        i=0
        each_y_value=[]#每个组里面一个是一个源点的样本
        info=""
        for item in batch:
            if i==0:
               info=item[0]#源点名字
               each_y_value.append(item[1])#源点数据
            else:
               if(str(info).__eq__(item[0])):
                  each_y_value.append(item[1])

               else:
                  ys.append([info,each_y_value])
                  each_y_value=[]
                  info=item[0]#源点名
                  each_y_value.append(item[1])#源点数据
            i=i+1
        ys.append([info,each_y_value])
        return ys,i #[原点名字，list[数据]]

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
                    batch_ys,total_length=feed_dict_fence(tf_feed.next_batch(batch_size))
                    list_length_first=batch_ys.__len__()
                    print(" list_length_first_fly:====",list_length_first)

                    # for jjj in range(list_length_first):
                    #     print(batch_ys[jjj][0])

                    if  list_length_first>2:#如果对比的源点数据少于3个无法进行判别
                        info_N=np.zeros([list_length_first,list_length_first])
                        info_distance=np.zeros([list_length_first,list_length_first])

                        pearson_out_module = tf.load_op_library('/tensorflow_user_lib/pearson_out.so')
                        temp_shape=tf.zeros([1])#传递shape的参数
                        #modle_type=tf.zeros([1])#传递使用的距离模型 1欧式距离 2Pearson距离 3Manhattan 4余旋距离

                        def ttf_k(x):#快速tff变换
                            ttf=[]
                            rezult=np.fft.fft(x)
                            # max_r=max([float(e.real) for e in rezult])
                            # min_r=min([float(e.real) for e in rezult])
                            #
                            # max_i=max([float(e.imag) for e in rezult])
                            # min_i=min([float(e.imag) for e in rezult])
                            #
                            # total_max=(max_r+max_i)/2
                            # total_max=(min_r+min_i)/2

                            for i in [[float(e.real),float(e.imag)] for e in rezult]:
                                if np.abs(i[0])<0.05:
                                    ttf.append(0)
                                else:
                                    ttf.append(i[0])

                                if np.abs(i[1])<0.05:
                                    ttf.append(0)
                                else:
                                    ttf.append(i[1])
                            return np.asarray(ttf[0:int(ttf.__len__()/2)])
                            # return np.asarray(ttf)

                        def smooth(x,len):
                            x_list=np.array(x)
                            pitch=int(x_list.size/len)
                            # yu=x_list.size%len
                            results=[]
                            for i in range(pitch):
                                results.append(np.average(x_list[i:i*len+len]))
                            return np.array(results)

                        batch_ys=[[e[0],smooth(ttf_k(e[1]),10)] for e in batch_ys]

                        #求最大的最小值和最小的最大值，用于判断是否输出为异常
                        batch_max=[[e[0],max(abs(e[1]))] for e in batch_ys]
                        batch_max_out=[]
                        for max_i in range(batch_max.__len__()):
                            max_avr=-1
                            for max_j in range(batch_max.__len__()):
                                if max_i!=max_j:
                                     value=batch_max[max_i][1]/batch_max[max_j][1]
                                     if max_avr<0 or value<max_avr:
                                        max_avr=value
                            batch_max_out.append(max_avr)

                        avg_var_max=np.average(np.array(batch_max_out))#均值
                        ad_var_max=np.std(np.array(batch_max_out))#标准差

                        #print("max:======", batch_max_out)
                        print("max异常值范围:======",avg_var_max-1.8*ad_var_max,avg_var_max+1.8*ad_var_max)
                        a_i=0
                        b_i=0
                        with tf.Session() as sess:
                            for i in range(list_length_first):
                                batch_len_a=batch_ys[i][1].__len__()
                                for j in range(list_length_first):
                                    if(i!=j):
                                      batch_len_b=batch_ys[j][1].__len__()
                                      if batch_len_a>batch_len_b:
                                         a_i=np.array(batch_ys[i][1][0:batch_len_b])
                                         batch_len_a=batch_len_b
                                         b_i=batch_ys[j][1]
                                      else:
                                          if batch_len_a<batch_len_b:
                                             b_i=np.array(batch_ys[j][1][0:batch_len_a])
                                             a_i=batch_ys[i][1]
                                          else:
                                              a_i=batch_ys[i][1]
                                              b_i=batch_ys[j][1]
                                      info_N[i][j]= sess.run(pearson_out_module.pearson_out(a_i,b_i,sess.run(temp_shape),np.array([2.0],dtype=float)))
                                      print("one:=%s,two=%s,r=%f"%(batch_ys[i][0],batch_ys[j][0],info_N[i][j]))
                                      # info_distance[i][j]= sess.run(pearson_out_module.pearson_out(a_i,b_i,sess.run(temp_shape),np.array([1.0],dtype=float)))
                                      # print("distance:=%s,two=%s,r=%f"%(batch_ys[i][0],batch_ys[j][0],info_distance[i][j]))
                        sess.close()

                        #计算2倍标准差之内的点
                        # avg_list=[]
                        # for i in range(list_length_first):
                        #     for j in range(list_length_first):
                        #         if j>i:
                        #            avg_list.append(info_distance[i][j])
                        #
                        # avg_var=np.average(np.array(avg_list))#均值
                        # ad_var=np.std(np.array(avg_list))#标准差
                        #
                        # info_order_distance=np.zeros([list_length_first,list_length_first])
                        info_order=np.zeros([list_length_first,list_length_first])#计算相关系数的排序，越小位数越大
                        for i in range(list_length_first):
                            for j in range(list_length_first):
                                index=list_length_first
                                # print("index:=",index)
                                # index_distance=list_length_first
                                if j==i:
                                   info_order[i][j]=-1
                                   # info_order_distance[i][j]=-1
                                else:
                                   for w in range(list_length_first):
                                       if w!=i:
                                         if info_N[i][j]>info_N[w][j] or info_N[i][j]>=0.90:
                                            index=index-1
                                         # if info_distance[i][j]<info_distance[w][j]:
                                         #    index_distance=index_distance-1
                                   info_order[i][j]=index
                                   # info_order_distance[i][j]=index_distance

                        print("输出order！====：")
                        for i in range(list_length_first):
                            for j in range(list_length_first):
                                print("order:=%s,j=%d,r=%f"%(batch_ys[i][0],j,info_order[i][j]))
                                #print("order_distance:=%s,j=%d,r=%f"%(batch_ys[i][0],j,info_order_distance[i][j]))

                        #属于异常值的规则
                        if list_length_first>=5 and list_length_first<=7:
                           for i in range(list_length_first):
                                mark_list=[]
                                mark_list_distance=[]
                                for j in range(list_length_first):
                                    if info_order[i][j]>=list_length_first-1:
                                       mark_list.append(1)#相关性排在倒数1位以内
                                    # if info_order_distance[i][j]>=list_length_first:
                                    #    mark_list_distance.append(1)#相关性排在倒数1位以内

                                print("每个的次序量=",batch_ys[i][0],sum(mark_list))
                                # self_distance=np.average([e for e in info_order_distance[i] if e!=-1])

                                continue_N=0
                                #相关系数检查结果不用重复判断直接判错
                                if  sum(mark_list)>=list_length_first-1 and abs(np.average(batch_ys[i][1]))>0.05:#如果当前源点和其他源点的相关系数排位在倒数二位以内的占比低于占到了全部点的
                                    results.append(batch_ys[i][0])
                                    print("成功放入！------------",batch_ys[i][0])
                                    continue_N=1

                                if  continue_N==0 and batch_max_out[i]>avg_var_max+1.8*ad_var_max or batch_max_out[i]<avg_var_max-1.8*ad_var_max and abs(np.average(batch_ys[i][1]))>0.05:
                                #进行积分重合度测试判断是否最后放入到异常数据
                                    print("放入检查！------------",batch_ys[i][0])
                                    #a=(x'*y+y'*x)/y‘*y,如何a的值比较大认为存在是异常点
                                    jf_list=np.zeros([list_length_first,list_length_first])
                                    for jf_i in range(list_length_first):
                                        a_jf=np.array(batch_ys[jf_i][1])
                                        jf_d=np.dot(np.transpose(a_jf),a_jf)
                                        a_length=a_jf.__len__()
                                        for jf_j in range(list_length_first):#用a*a_jf去逼近b_jf
                                            if jf_j!=jf_i:
                                                b_jf=np.array(batch_ys[jf_j][1])
                                                b_length=b_jf.__len__()
                                                if a_length<b_length:
                                                   b_jf=np.array(batch_ys[jf_j][1][0:a_length])
                                                else:
                                                    if a_length>b_length:
                                                       a_jf=np.array(batch_ys[jf_i][1][0:b_length])
                                                       a_length=a_jf.__len__()
                                                       jf_d=np.dot(np.transpose(a_jf),a_jf)
                                                    else:
                                                        pass
                                                if abs(jf_d)>0.01:
                                                    jf_list[jf_i][jf_j]=np.dot(np.transpose(a_jf),b_jf)/jf_d
                                                else:
                                                    jf_list[jf_i][jf_j]=np.dot(np.transpose(a_jf),b_jf)/(0.01)
                                            else:
                                                jf_list[jf_i][jf_j]=0

                                    print("jf_list[jf_i][jf_j]==",jf_list)
                                    #按列求均值
                                    avg_jf=[]
                                    jf_list=np.array(jf_list)
                                    for jf_i in range(list_length_first):
                                        avg_jf.append(np.average(np.array([e for e in np.array(jf_list)[:,jf_i] if e!=0])))
                                    print("a的均值：========",avg_jf)
                                    av_jf=np.average(np.array(avg_jf))#均值
                                    ad_jf=np.std(np.array(avg_jf))#标准差
                                    if avg_jf[i]>av_jf+2.0*ad_jf or avg_jf[i]<av_jf-2.0*ad_jf:
                                       results.append(batch_ys[i][0])
                                       print("成功放入！------------",batch_ys[i][0])
                                    #4分之3以上怀疑为异常点
                                    # (sum(mark_list_distance)>=list_length_first-1 and (self_distance<avg_var-1.0*ad_var or self_distance>avg_var+1.0*ad_var \
                                    #                                                    )))
                        else:
                            if list_length_first>7:
                                    for i in range(list_length_first):
                                       mark_list=[]
                                       mark_list_distance=[]
                                       for j in range(list_length_first):
                                           if info_order[i][j]>=list_length_first-1:
                                              mark_list.append(1)#相关性排在倒数1位以内
                                           # if info_order_distance[i][j]>=list_length_first:
                                           #    mark_list_distance.append(1)#相关性排在倒数1位以内
                                       print("每个的次序量=",batch_ys[i][0],sum(mark_list))
                                       #self_distance=np.average([e for e in info_order_distance[i] if e!=-1])

                                       continue_N=0
                                       #相关系数检查结果不用重复判断直接判错
                                       if  sum(mark_list)>=list_length_first-2 and abs(np.average(batch_ys[i][1]))>0.05:#如果当前源点和其他源点的相关系数排位在倒数二位以内的占比低于占到了全部点的
                                           results.append(batch_ys[i][0])
                                           print("成功放入！------------",batch_ys[i][0])
                                           continue_N=1

                                       if  continue_N==0 and batch_max_out[i]>avg_var_max+1.8*ad_var_max or batch_max_out[i]<avg_var_max-1.8*ad_var_max and abs(np.average(batch_ys[i][1]))>0.05:
                                           #进行积分重合度测试判断是否最后放入到异常数据
                                           print("放入检查！------------",batch_ys[i][0])
                                           #a=(x'*y+y'*x)/y‘*y,如何a的值比较大认为存在是异常点
                                           jf_list=np.zeros([list_length_first,list_length_first])
                                           for jf_i in range(list_length_first):
                                               a_jf=np.array(batch_ys[jf_i][1])
                                               jf_d=np.dot(np.transpose(a_jf),a_jf)
                                               a_length=a_jf.__len__()
                                               for jf_j in range(list_length_first):#用a*a_jf去逼近b_jf
                                                   if jf_j!=jf_i:
                                                       b_jf=np.array(batch_ys[jf_j][1])
                                                       b_length=b_jf.__len__()
                                                       if a_length<b_length:
                                                           b_jf=np.array(batch_ys[jf_j][1][0:a_length])
                                                       else:
                                                           if a_length>b_length:
                                                               a_jf=np.array(batch_ys[jf_i][1][0:b_length])
                                                               a_length=a_jf.__len__()
                                                               jf_d=np.dot(np.transpose(a_jf),a_jf)
                                                           else:
                                                               pass
                                                       if abs(jf_d)>0.01:
                                                           jf_list[jf_i][jf_j]=np.dot(np.transpose(a_jf),b_jf)/jf_d
                                                       else:
                                                           jf_list[jf_i][jf_j]=np.dot(np.transpose(a_jf),b_jf)/(0.01)
                                                   else:
                                                       jf_list[jf_i][jf_j]=0

                                           print("jf_list[jf_i][jf_j]==",jf_list)
                                           #按列求均值
                                           avg_jf=[]
                                           jf_list=np.array(jf_list)
                                           for jf_i in range(list_length_first):
                                               avg_jf.append(np.average(np.array([e for e in np.array(jf_list)[:,jf_i] if e!=0])))
                                           print("a的均值：========",avg_jf)
                                           av_jf=np.average(np.array(avg_jf))#均值
                                           ad_jf=np.std(np.array(avg_jf))#标准差
                                           if avg_jf[i]>av_jf+2.0*ad_jf or avg_jf[i]<av_jf-2.0*ad_jf:
                                               results.append(batch_ys[i][0])
                                               print("成功放入！------------",batch_ys[i][0])
                                               #4分之3以上怀疑为异常点
                                               # (sum(mark_list_distance)>=list_length_first-1 and (self_distance<avg_var-1.0*ad_var or self_distance>avg_var+1.0*ad_var \
                                               #                                                    )))

                            else:#如果样本点少于等于3个
                               for i in range(list_length_first):
                                   mark_list=[]
                                   mark_list_distance=[]
                                   for j in range(list_length_first):
                                       if info_order[i][j]>=list_length_first-1:
                                          mark_list.append(1)#相关性排在倒数1位以内
                                       # if info_order_distance[i][j]>=list_length_first:
                                       #    mark_list_distance.append(1)#相关性排在倒数1位以内
                                   print("每个的次序量=",batch_ys[i][0],sum(mark_list))
                                   # self_distance=np.average([e for e in info_order_distance[i] if e!=-1])

                                   continue_N=0
                                   #相关系数检查结果不用重复判断直接判错
                                   if  sum(mark_list)>=list_length_first-1 and max(info_N[i])<0.30 and abs(np.average(batch_ys[i][1]))>0.05:#如果当前源点和其他源点的相关系数排位在倒数二位以内的占比低于占到了全部点的
                                       results.append(batch_ys[i][0])
                                       print("成功放入！------------",batch_ys[i][0])
                                       continue_N=1

                                   if  continue_N==0 and batch_max_out[i]>avg_var_max+2.0*ad_var_max or batch_max_out[i]<avg_var_max-2.0*ad_var_max and abs(np.average(batch_ys[i][1]))>0.05:
                                       #进行积分重合度测试判断是否最后放入到异常数据
                                       print("放入检查！------------",batch_ys[i][0])
                                       #a=(x'*y+y'*x)/y‘*y,如何a的值比较大认为存在是异常点
                                       jf_list=np.zeros([list_length_first,list_length_first])
                                       for jf_i in range(list_length_first):
                                           a_jf=np.array(batch_ys[jf_i][1])
                                           jf_d=np.dot(np.transpose(a_jf),a_jf)
                                           a_length=a_jf.__len__()
                                           for jf_j in range(list_length_first):#用a*a_jf去逼近b_jf
                                               if jf_j!=jf_i:
                                                   b_jf=np.array(batch_ys[jf_j][1])
                                                   b_length=b_jf.__len__()
                                                   if a_length<b_length:
                                                       b_jf=np.array(batch_ys[jf_j][1][0:a_length])
                                                   else:
                                                       if a_length>b_length:
                                                           a_jf=np.array(batch_ys[jf_i][1][0:b_length])
                                                           a_length=a_jf.__len__()
                                                           jf_d=np.dot(np.transpose(a_jf),a_jf)
                                                       else:
                                                           pass
                                                   if abs(jf_d)>0.01:
                                                       jf_list[jf_i][jf_j]=np.dot(np.transpose(a_jf),b_jf)/jf_d
                                                   else:
                                                       jf_list[jf_i][jf_j]=np.dot(np.transpose(a_jf),b_jf)/(0.01)
                                               else:
                                                   jf_list[jf_i][jf_j]=0

                                       print("jf_list[jf_i][jf_j]==",jf_list)
                                       #按列求均值
                                       avg_jf=[]
                                       jf_list=np.array(jf_list)
                                       for jf_i in range(list_length_first):
                                           avg_jf.append(np.average(np.array([e for e in np.array(jf_list)[:,jf_i] if e!=0])))
                                       print("a的均值：========",avg_jf)
                                       av_jf=np.average(np.array(avg_jf))#均值
                                       ad_jf=np.std(np.array(avg_jf))#标准差
                                       if avg_jf[i]>av_jf+2.0*ad_jf or avg_jf[i]<av_jf-2.0*ad_jf:
                                           results.append(batch_ys[i][0])
                                           print("成功放入！------------",batch_ys[i][0])
                                           #4分之3以上怀疑为异常点
                                           # (sum(mark_list_distance)>=list_length_first-1 and (self_distance<avg_var-1.0*ad_var or self_distance>avg_var+1.0*ad_var \
                                           #                                                    )))

                        num_lack=total_length-results.__len__()
                        if num_lack>0:
                            results.extend([["o","o","o"]]*num_lack)
                        tf_feed.batch_results(results)
                    else:
                        print("最后一个了！")
                        results=[]
                        tf_feed.batch_results(results)
                    i_n=i_n+1
            tf_feed.terminate()