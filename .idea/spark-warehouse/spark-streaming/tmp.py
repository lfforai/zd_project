
import tensorflow as tf
batch_ys = range(100)
list_length=batch_ys.__len__()
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
# T = tf.placeholder(dtype=tf.float32, shape=(None)) #状态参数
# Q = tf.placeholder(dtype=tf.float32, shape=(None)) #状态偏差
# T = tf.placeholder(dtype=tf.float32, shape=(None)) #状态参数
# Q = tf.placeholder(dtype=tf.float32, shape=(None)) #状态偏差
global_step = tf.Variable(0,dtype=tf.int64,trainable=False)

array_list=[]
array_list1=[]
var_list=[T,Z,H,Q,d]

for i in range(batch_ys.__len__()):
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
print("y",y_st_sum)
F_sum=tf.stack(array_list1,axis=-1)
loss=-(tf.reduce_sum(tf.log(tf.abs(F_sum)))/2+tf.reduce_sum((y-y_st_sum)*(1/F_sum)*(y-y_st_sum))/2)
train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001,rho=0.85).minimize(
    1-loss, global_step=global_step)
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    print(sess.run(loss,feed_dict={y:batch_ys}))
    for i in range(int(2500)):
        # sess.run(tf.initialize_all_variables())
        sess.run(train_op,feed_dict={y:batch_ys})
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    # saver.restore(sess,"/tmp/my-model")
    print(sess.run(loss,feed_dict={y:batch_ys}))
    print(sess.run(H))
sess.close()

