
import numpy as np

def normal_probability(y,n=50000,p=0.95,gpu_num="0"):
    #用反复逼近的方式迭代求最接近最大值的
    import numpy as np
    import tensorflow as tf

    with tf.device("/cpu:"+str(gpu_num)):
        # y_in=tf.placeholder(dtype=tf.float32,shape=(None))
        def normal_probability_density(x_s):
            #均值=0,标准差=h的正态分布的概率密度函数
            pi=3.141592654
            h=1.0/np.power(y.__len__(),0.2)
            result=np.mean(-np.power(x_s-y,2)/(np.power(h,2)*2.0)/(np.power(pi*2.0,0.5)*h))
            return result

        value_big=tf.get_variable("value_big",shape=[],dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer)
        value_little=tf.get_variable("value_little",shape=[],dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer)
        value_now=tf.get_variable("value_now",shape=[],dtype=tf.float32,trainable=False,initializer=tf.zeros_initializer)
        config = tf.ConfigProto()#luofeng jia
        config.gpu_options.allow_growth=True
        #赋值
        value_big=tf.assign(value_big,tf.convert_to_tensor(y.max()+100,dtype=tf.float32))
        value_little=tf.assign(value_little,tf.convert_to_tensor(y.min()-100,dtype=tf.float32))
        value_now=tf.assign(value_now,value_big)

        min_cast=tf.convert_to_tensor(y.min()-100,dtype=tf.float32)#下限
        dx=(value_now-value_little)/n

        list_dx=tf.convert_to_tensor(np.linspace(1,n,n),dtype=tf.float32)
        x1=min_cast+list_dx*dx

        p_now_np=0
        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            sess.run(local_init_op)
            rdd_dataset= tf.contrib.data.Dataset.from_tensor_slices(x1)\
                .map(normal_probability_density).map(lambda x2:x2*dx)

              #.map(lambda x:min_cast+tf.cast(x,dtype=tf.float32)*dx).map(normal_probability_density(y_in,-1,y.__len__())).map(lambda x:x*dx))
            p_now_np=sess.run(p_now)
            if p_now_np-p<0.005 and p_now_np-p>-0.005:
                rezult=sess.run(value_now)
            else:
                if p_now_np>p:#big保留,little调整
                    value_big=value_now
                    value_now=(value_big-value_little)/2
                else:
                    value_little=value_now
                    value_now=(value_big-value_little)/2
    return p_now_np,rezult

y=np.random.normal(loc=0.0, scale=0.5, size=1000)
normal_probability(y,n=50000,p=0.95,gpu_num="0")
# normal_probability(min,max,y,n=50000,p=0.95,gpu_num="0")