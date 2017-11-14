
import numpy as np

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

            rdd_dataset=tf.contrib.data.Dataset.from_tensor_slices(x1).map(normal_probability_density(y_in,dx/2),num_threads=16).map(lambda x2:x2*dx,num_threads=16)

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

y=np.random.normal(loc=0.0, scale=1, size=300)
print(1/np.power(5000,0.2))
p,p_value=normal_probability(y,n=2000,p=0.85,gpu_num="0")
print("概率：=%f，分位值：=%f"%(p,p_value))

# normal_probability(min,max,y,n=50000,p=0.95,gpu_num="0")