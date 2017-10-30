#-*- coding:utf-8 -*-
import tensorflow as tf
import sys
#创建稍微复杂一点的队列作为例子
q = tf.FIFOQueue(1000,"float")
#计数器
counter = tf.Variable(0.0)
#操作:给计数器加一
increment_op = tf.assign_add(counter,tf.constant(1.0))
enqueue_op = q.enqueue(counter) # 操作：计数器值加入队列
#操作:将计数器加入队列
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)

# 主线程
sess = tf.Session()
sess.run(tf.initialize_all_variables())

#Coordinator:协调器,协调线程间的关系,可以视为一种信号量,用来做同步
coord = tf.train.Coordinator()

## 启动入队线程, Coordinator是线程的参数
enqueue_threads = qr.create_threads(sess, coord = coord,start=True)  # 启动入队线程

# 主线程
for i in range(0, 10):
    print("-------------------------")
    print(sess.run(q.dequeue()))

    #通知其他线程关闭
coord.request_stop()
#其他所有线程关闭之后，这一函数才能返回
#join操作经常用在线程当中,其作用是等待某线程结束
coord.join(enqueue_threads)