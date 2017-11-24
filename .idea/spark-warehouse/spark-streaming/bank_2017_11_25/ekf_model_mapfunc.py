from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

def print_log(worker_num, arg):
    print("{0}: {1}".format(worker_num, arg))

def get_available_gpus_len():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU'].__len__()


def map_func_ekf(args, ctx):
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
                # print(item)
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


    def feed_dict_fence(batch):
        # Convert from [(images, labels)] to two numpy arrays of the proper type
        list_xs_1=[]
        partitionnum=0
        y = []
        i=0
        for item in batch:
            if(i==0):
                # print(item)
                partitionnum=item[0]
                y.append(item[1][1])
                list_xs_1.append(item[1][0])
                i=i+1
            else:
                y.append(item[1][1])
                list_xs_1.append(item[1][0])

        ys = numpy.array(y)
        ys = ys.astype(numpy.float32)
        xs=numpy.array(range(ys.__len__()))
        xs=xs.astype(numpy.float32)
        return partitionnum,(list_xs_1,xs,ys)

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
                for i in range(args.steps):
                    if i==0:
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
                        if(tf_feed.should_stop()):
                                tf_feed.terminate()
                        print("--------------------第"+str(ctx.task_index)+"task的第"+str(i+1)+"步迭代---------------------------------")
                        num,(batch_xs, batch_ys) = feed_dict(tf_feed.next_batch(batch_size))
                        if marknum==0 or not str(num).__eq__(p_num):
                            # logdir = TFNode.hdfs_path(ctx,str("model/")+args.model+str("_{0}/").format(num))
                            # logdir=logdir.replace("127.0.0.1:9000","sjfx1:9000")
                            logdir = "hdfs://sjfx1:9000/"+str("model/")+args.model+str("_{0}").format(num)
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
                            sess.run(init_op)
                            sess.run(local_init_op)
                            print("before ptimizer:loss=",sess.run(loss,feed_dict={y:batch_ys}))
                            for n in range(int(100)):
                                # sess.run(tf.initialize_all_variables())
                                sess.run(train_op,feed_dict={y:batch_ys})
                            save_path = saver.save(sess, logdir)
                            print("Model saved in file: %s" % save_path)
                            # saver.restore(sess,"/tmp/my-model")
                            print("after ptimizer:loss=",sess.run(loss,feed_dict={y:batch_ys}))
                            print("H:=",sess.run(H))
                        sess.close()
                    else:
                        tf.reset_default_graph()
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
                        if(tf_feed.should_stop()):
                           tf_feed.terminate()
                        print("--------------------第"+str(ctx.task_index)+"task的第"+str(i+1)+"步迭代---------------------------------")
                        num,(batch_xs, batch_ys) = feed_dict(tf_feed.next_batch(batch_size))
                        if marknum==0 or not str(num).__eq__(p_num):
                            # logdir = TFNode.hdfs_path(ctx,str("model/")+args.model+str("_{0}/").format(num))
                            logdir = "hdfs://sjfx1:9000/"+str("model/")+args.model+str("_{0}").format(num)
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
                            saver.restore(sess,logdir)
                            print("Model restored.")
                            print("before H:=",sess.run(H))
                            print("before ptimizer:loss=",sess.run(loss,feed_dict={y:batch_ys}))
                            for n in range(int(100)):
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

            else:#测试"inference"
                # Add ops to save and restore all the variables.
                while not tf_feed.should_stop():
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

                    num,(xs_info,batch_xs,batch_ys)= feed_dict_fence(tf_feed.next_batch(batch_size))
                    list_length=batch_ys.__len__()
                    print("length:=========",list_length)
                    data = {
                        tf.contrib.timeseries.TrainEvalFeatures.TIMES:batch_xs,
                        tf.contrib.timeseries.TrainEvalFeatures.VALUES:batch_ys,
                    }

                    if marknum==0 or not str(num).__eq__(p_num):
                       # logdir = TFNode.hdfs_path(ctx,str("model/")+args.model+str("_{0}/").format(num))
                       # logdir=logdir.replace("127.0.0.1:9000","sjfx1:9000")
                       logdir = "hdfs://sjfx1:9000/"+str("model/")+args.model+str("_{0}").format(num)
                       marknum=marknum+1
                       p_num=num
                       print("logdir================:",logdir)

                    init_op = tf.global_variables_initializer()
                    local_init_op = tf.local_variables_initializer()
                    result=[]
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
                                # sess.run([a_t_t_1, p_t_t_1,F,a_t,p_t_1])
                                # rep=sess.run(Z*a_t+d)
                                # result.append((batch_ys[i],rep,batch_ys[i]-rep))
                            else:
                                a_t_t_1=tf.assign(a_t_t_1,T*a_t+c)#1
                                p_t_t_1=tf.assign(p_t_t_1,T*p_t_1*T+Q)#2

                                F=Z*p_t_t_1*Z+H#3
                                a_t=tf.assign(a_t,a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys[i]-Z*a_t_t_1-d))#4
                                p_t_1=tf.assign(p_t_1,p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1)#5
                                #预测的y_st
                                # sess.run([a_t_t_1, p_t_t_1,F,a_t,p_t_1])
                                rep=sess.run(Z*a_t+d)
                                # print("rep=======:",rep)
                                result.append([batch_ys[i],rep,batch_ys[i]-rep])
                        result =[[num,e[0],e[1],e[2],l] for e,l in zip(result,xs_info)]
                        num_lack=list_length-result.__len__()
                        if num_lack>0:
                            result.extend([["o","o","o"]]*num_lack)
                        tf_feed.batch_results(result)
                    # sess.close()
            tf_feed.terminate()