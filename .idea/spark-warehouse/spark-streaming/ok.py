import tensorflow as tf
import numpy
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
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    # Check the values of the variables
    print("T : %s" % T.eval())
    print("Z : %s" % Z.eval())
    print("H : %s" % H.eval())
    print("Q : %s" % Q.eval())
    print("d : %s" % d.eval())
    print("c : %s" % c.eval())
    batch_ys = range(1000)
    for i in range(batch_ys.__len__()):
        if i==0:
            a_t_t_1=tf.assign(a_t_t_1,T*batch_ys[0]+c)#1
            p_t_t_1=tf.assign(p_t_t_1,T*Q*T+Q)#2

            F=Z*p_t_t_1*Z+H#3
            a_t=tf.assign(a_t,a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys[0]-Z*a_t_t_1-d))#4
            p_t_1=tf.assign(p_t_1,p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1)#5
            #预测的y_st
            print(sess.run(Z*a_t+d))

        else:
            a_t_t_1=tf.assign(a_t_t_1,T*a_t+c)#1
            p_t_t_1=tf.assign(p_t_t_1,T*p_t_1*T+Q)#2

            F=Z*p_t_t_1*Z+H#3
            a_t=tf.assign(a_t,a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys[i]-Z*a_t_t_1-d))#4
            p_t_1=tf.assign(p_t_1,p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1)#5
            #预测的y_st
            print(sess.run(Z*a_t+d))


print(numpy.array([1,2,3])/numpy.array([2])*numpy.array([1,2,3]))