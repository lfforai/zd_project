import re
import numpy as np
import jieba

#
s="金紫山一期25号风机3#变桨电容柜体温度(℃)"
seg_list = jieba.cut(s, cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut(s, cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut(s)  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search(s)  # 搜索引擎模式
print(", ".join(seg_list))
#
# 沈阳驼山一期风机22.1#变桨速率 G_DBTS_SY_01W_022FJ_CN004
# [0, 0, 7787, 7907, 7918, 7932]

import csv
device_code=['GU','GS','GT','GA','GW','HJ','GL','GR','FJ']#9
d_l=device_code.__len__()

state_code=['CD','PW','PR','PP','PJ','JR','CN','CU','PF','PV'\
            ,'PA','UR','TM','MS','SS','BF','MA','ST','TS','SA','AD'\
            ,'CA','CT','CP','XD','EF','RR','WP','YD','TP','TR'\
            ,'AN','EN','RA','EX','ZT','ZJ','VA','ZK','ZD','YC','BS','BJ']#43
s_l=state_code.__len__()

for i in range(s_l-d_l):
    device_code.append("wwww")
d_l=device_code.__len__()
# matchObj=re.search(r'\.\d+', "adsfasc23", re.M|re.I)
# if matchObj:
#    print(matchObj.group())
# exit()

def text2jieba(temp):
    temp=str(temp) \
        .replace("℃","") \
        .replace("L1","").replace("L2","").replace("L3","").replace("I1","").replace("I2","") \
        .replace("I2","").replace("#","").replace(" ","").replace("]","_") \
        .replace("[","_").replace("(","_").replace(")","_").replace("（","_") \
        .replace("）","_")
    # matchObj= re.search( r'\(.*\)', temp, re.M|re.I)
    # if matchObj:
    #      print("search --> matchObj.group() : ", matchObj.group())
    #      temp=temp.replace(matchObj.group(),"")
    # else:
    #      pass
    #
    # matchObj= re.search( r'（.*）', temp, re.M|re.I)
    # if matchObj:
    #     print("search --> matchObj.group() : ", matchObj.group())
    #     temp=temp.replace(matchObj.group(),"")
    # else:
    #     pass
    #
    # matchObj= re.search(r'\[.*\]', temp, re.M|re.I)
    # if matchObj:
    #      print("search --> matchObj.group() : ", matchObj.group())
    #      temp=temp.replace(matchObj.group(),"")
    # else:
    #      pass

    matchObj= re.search(r'\.\d+', temp, re.M|re.I)
    if matchObj:
        # print("search --> matchObj.group() : ", matchObj.group())
        temp=temp.replace(matchObj.group(),"")
    else:
        pass

    is_digt=False
    seg_list = jieba.cut_for_search(temp)
    cc=list(set(",".join(seg_list).replace("#","").replace("_","") \
        .split(",")))

    def mapfunc(x,is_digt=is_digt):
        if str(x).__eq__('.'):
            return ""
        if str(x).isdigit() and is_digt==False:
            is_digt=True
            return x
        else:
            if str(x).isdigit() and is_digt==True:
                return ""
            else:
                return x

    cc=list(map(mapfunc,cc))
    cc=list(filter(lambda x:str(x)!="",cc))
    return cc

# print("luofeng",text2jieba(s))
# exit()
#text分解到one_hot向量,len返回的固定向量长度,10
def text2onehot(text,marking,std_one_hot=[],device_code=device_code,state_code=state_code,len=10):
    global is_digt
    device_code_len=d_l
    state_code_len=s_l
    result_mark=list(np.zeros(device_code_len+state_code_len))
    text_list=list(text2jieba(text))#分解为
    # 景峡一期每日发电量 := G_XJJX_HM_01W_001GL_PJ001
    # ['景峡', '一期', '每日', '发电量']
    result=[int(list(std_one_hot).index("_"))]*text_list.__len__() #初始化·["_","_","_"]
    i=0
    for e in text_list:
        if std_one_hot.__contains__(e):
            index_N=int(list(std_one_hot).index(e))
            result[i]=index_N
            i=i+1
        else:
            index_N=int(list(std_one_hot).index("_"))
            result[i]=index_N
            i=i+1
    result=sorted(result,reverse=True)
    result_len=result.__len__()
    if result_len>len:
       result=result[0:len]#如果超过len截断前面的数据只保留后面
    else:
       if result_len<len:
          result.extend([int(list(std_one_hot).index("_"))]*(len-result_len))
       else:
          pass

    marking_list=str(marking).replace(" ","")[-8:-3].split("_")
    if device_code.__contains__(str(marking_list[0])) and state_code.__contains__(str(marking_list[1])) :
       result_mark[device_code.index(str(marking_list[0]))]=1
       result_mark[state_code.index(str(marking_list[1]))+device_code_len]=1
       return [result,result_mark]#one_hot向量和所属标记
    else:
       return ['bad','bad']

#把字符串输入，转化为[0,0,0,1,0]的onehot
def text2onehot_2(text,marking,std_one_hot=[],device_code=device_code,state_code=state_code,std_one_hot_len=1849):
    global is_digt
    device_code_len=d_l
    state_code_len=s_l
    result_mark=list(np.zeros(device_code_len+state_code_len))
    text_list=list(text2jieba(text))#分解为
    # 景峡一期每日发电量 := G_XJJX_HM_01W_001GL_PJ001
    # ['景峡', '一期', '每日', '发电量']
    result=[0]*std_one_hot_len #初始化·["_","_","_"]
    index_H=int(list(std_one_hot).index("_"))

    for e in text_list:
        if std_one_hot.__contains__(e):
            index_N=int(list(std_one_hot).index(e))
            result[index_N]=index_N
        else:
            result[index_H]=index_H

    marking_list=str(marking).replace(" ","")[-8:-3].split("_")
    if device_code.__contains__(str(marking_list[0])) and state_code.__contains__(str(marking_list[1])) :
        result_mark[device_code.index(str(marking_list[0]))]=1
        result_mark[state_code.index(str(marking_list[1]))+device_code_len]=1
        return [result,result_mark]#one_hot向量和所属标记
    else:
        return ['bad','bad']

#line="济南生产运营中心庄子_风机10_11#DP故障信息(luoe)[ddd]"
def read2words(filename='/PointData_201801051031.csv'):
    print("开始建立word词库！")
    bid_info = csv.DictReader(open(filename,'r'))
    dict_data = []
    for lines in bid_info:
        if bid_info.line_num == 1:
            continue
        else:
            dict_data.append(lines)
    row_num = len(dict_data)
    # print('this is all the data---' + str(dict))

    #循环读取每一行
    i = 0
    j = 0
    total_space={"_":0}
    while(i < row_num):
        if not str(dict_data[i]["name"]).__eq__(""):
           #print(str(dict_data[i]["z"]),":=",dict_data[i]["name"])
           cc=text2jieba(dict_data[i]["z"])
           #print(cc)
           #print("----------------------------------------------")
           for e in cc:
                if  total_space.get(e):
                    total_space[e]=total_space[e]+1
                else:
                    total_space[e]=1
           j=j+1
        i += 1
    # print("row_num:=",row_num)
    # print("row_j:=",j)
    total_space["_"]=1
    b=zip(total_space.keys(),total_space.values())   #拉成Tuple对组成的List
    total_space=list(sorted(b, key=lambda item:item[1]))
    total_space=dict(filter(lambda x: True if int(x[1])>10 else False,total_space))
    std_one_hot=total_space.keys()#标准化词汇表
    # print(std_one_hot)
    print("word词库已经建立完成")
    return list(std_one_hot)

#把marking [1......0.....1.....0] 转为 GL_PJ
def onehot2mark(result_mark=device_code,device_code=device_code,state_code=state_code):
    mark=[]
    result=list(np.zeros(2))
    for i in range(result_mark.__len__()):
        if result_mark[i]==1:
           mark.append(i)
        else:
           pass
    if mark.__len__()==2:
       result[0]=device_code[mark[0]]
       result[1]=state_code[mark[1]-device_code.__len__()]
    else:
       result="wrong"
    return str(result[0]+"_"+result[1])

#提取模型的x和y
def text2x_y(filename='/PointData_201801051031.csv',std_one_hot=[]):
    print("开始生成样本x_vs_y!")
    std_one_hot_len=list(std_one_hot).__len__()
    bid_info = csv.DictReader(open(filename,'r'))
    dict_data = []
    for lines in bid_info:
        if bid_info.line_num == 1:
            continue
        else:
            dict_data.append(lines)
    row_num = len(dict_data)
    # print('this is all the data---' + str(dict))
    result=[]
    #循环读取每一行
    i = 0
    j = 0
    total_space={"_":0}
    while(i < row_num):
        if not str(dict_data[i]["name"]).__eq__(""):
            #print(str(dict_data[i]["z"]),str(dict_data[i]["name"]))
            x,y=text2onehot(str(dict_data[i]["z"]),str(dict_data[i]["name"]),std_one_hot,len=std_one_hot_len)
            result.append([x,y])
            #print(x,y)
            #print("----------------------------------------------------")
            j=j+1
        i += 1
    print("结束生成样本x_vs_y!")
    return result

#提取模型的x和y
def text2x_y_2(filename='/PointData_201801051031.csv',std_one_hot=[]):
    print("开始生成样本x_vs_y!")
    std_one_hot_len=list(std_one_hot).__len__()
    bid_info = csv.DictReader(open(filename,'r'))
    dict_data = []
    for lines in bid_info:
        if bid_info.line_num == 1:
            continue
        else:
            dict_data.append(lines)
    row_num = len(dict_data)
    # row_num=1000
    # print('this is all the data---' + str(dict))
    result=[]
    #循环读取每一行
    i = 0
    j = 0
    total_space={"_":0}
    while(i < row_num):
        if not str(dict_data[i]["name"]).__eq__(""):
            #print(str(dict_data[i]["z"]),str(dict_data[i]["name"]))
            x,y=text2onehot_2(str(dict_data[i]["z"]),str(dict_data[i]["name"]),std_one_hot,std_one_hot_len=std_one_hot_len)
            result.append([x,y])
            #print(list(x).__len__())
            #print("@@@@@@")
            #print(list(y).__len__())
            #print("----------------------------------------------------")
            j=j+1
        i += 1
    print("结束生成样本x_vs_y!")
    return result

#创建词库
std_one_hot=read2words(filename='/PointData_201801051031.csv')
std_one_hot=std_one_hot[-1764:-1]
std_one_hot.append('_')
std_one_hot_len=std_one_hot.__len__()
power_std_one_hot_len=np.power(std_one_hot_len,0.5)
print(power_std_one_hot_len)
# print(text2jieba(s))
# a=text2onehot(s,"G_DBTS_SY_01W_022FJ_CN004",std_one_hot=std_one_hot)
# print(a)
# print(a[0][5],std_one_hot[a[0][5]])
# print(a[0][4],std_one_hot[a[0][4]])
# print(a[0][3],std_one_hot[a[0][3]])
# print(a[0][2],std_one_hot[a[0][2]])
# print(a[0][1],std_one_hot[a[0][1]])
# print(a[0][0],std_one_hot[a[0][0]])
# print(std_one_hot.index("风机"))
# exit()
# x,y=text2onehot("红星二场一期光伏日发电量"," G_XJTX_HM_01P_001GL_PJ001",std_one_hot)
# z=onehot2mark(result_mark=y)
# print(x)
# print(y)
# print(z)
#一、开始执行
#（1）创建样本储存为onehot-mark的格式
result=text2x_y_2(std_one_hot=std_one_hot)
result=list(filter(lambda x:True if not str(x[0]).__eq__("bad") else False,result))
# print(result)

#（2）tensoflow神经网
import tensorflow as tf

#相关参数设定
x_len=list(result[0][0]).__len__()
print("x_len:=",x_len)
y_len=list(result[0][1]).__len__()
print("y_len:=",y_len)
batchSize=10000

img=tf.convert_to_tensor(np.array([e[0] for e in result]),dtype=tf.float32)
label=tf.convert_to_tensor(np.array([e[1] for e in result]),dtype=tf.float32)

dataset_img = tf.contrib.data.Dataset.from_tensor_slices(img)
dataset_label = tf.contrib.data.Dataset.from_tensor_slices(label)
dataset = tf.contrib.data.Dataset.zip((dataset_img,dataset_label))

dataset = dataset.shuffle(buffer_size=100000)
dataset = dataset.batch(200)
dataset = dataset.repeat(20)
iterator = dataset.make_initializable_iterator()
next_element=iterator.get_next()

# with tf.Session() as sess:n
#     # sess.run(iterator.initializer)
#     for i in range(100):
#         x,y=sess.run(iterator.get_next())
#         print(x)
#         print("－－－－－－－－－－－－－－－",i)
# exit()

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, x_len])
ys = tf.placeholder(tf.float32, [None, y_len])
# 为了防止过拟合，在输出层之前加Dropout层
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("G", reuse=tf.AUTO_REUSE):
    # w1=tf.get_variable("w1", [20, 300])
    # b1=tf.get_variable("b1", [300])
    # w2=tf.get_variable("w2", [300, 150])
    # b2=tf.get_variable("b2", [150])
    # w3=tf.get_variable("w3", [150, 86])
    # b3=tf.get_variable("b3", [86])

    #卷积神经网
    def mlp_cnn(X,keep_prob,n_maxouts=3):
      with tf.device('/gpu:1'):
        #########卷积网络会有很多的权重和偏置需要创建，先定义好初始化函数以便复用########
        # 给权重制造一些随机噪声打破完全对称（比如截断的正态分布噪声，标准差设为0.1）
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        # 因为我们要使用ReLU，也给偏置增加一些小的正值（0.1）用来避免死亡节点（dead neurons）
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        ########卷积层、池化层接下来重复使用的，分别定义创建函数########
        # tf.nn.conv2d是TensorFlow中的2维卷积函数
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        # 使用2*2的最大池化
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

        # 使用3*3的最大池化
        def max_pool_3x3(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1], padding='SAME')
            ########正式设计卷积神经网络之前先定义placeholder########

        # x是特征，y_是真实label。将图片数据从1D转为2D。使用tensor的变形函数tf.reshape
        # x = tf.placeholder(tf.float32, shape=[None,X.get_shape()[1]])
        #y = tf.placeholder(tf.float32, shape=[None,Y.get_shape()[1]])
        x_image = tf.reshape(X,[-1,int(power_std_one_hot_len),int(power_std_one_hot_len),1])

        ########设计卷积神经网络########
        # 第一层卷积
        # 卷积核尺寸为5*5,1个颜色通道，32个不同的卷积核
        W_conv1 = weight_variable([5, 5, 1, 32])
        # 用conv2d函数进行卷积操作，加上偏置
        b_conv1 = bias_variable([32])
        # 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # 对卷积的输出结果进行池化操作
        h_pool1 = max_pool_2x2(h_conv1)
        #print("h_pool1.get_shape()",h_pool1.get_shape())

        # 第二层卷积（和第一层大致相同，卷积核为64，这一层卷积会提取64种特征）
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_3x3(h_conv2)
        # print("h_pool2.get_shape()",h_pool2.get_shape())
        # exit()

        # 全连接层。隐含节点数1024。使用ReLU激活函数
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # 为了防止过拟合，在输出层之前加Dropout层
        # keep_prob = tf.placeholder(tf.float32)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # 输出层。添加一个softmax层，就像softmax regression一样。得到概率输出。
        W_fc2 = weight_variable([1024, y_len])
        b_fc2 = bias_variable([y_len])
        fc4=tf.matmul(h_fc1, W_fc2) + b_fc2
        return fc4


    #全链接神经网，l1=input×300+300, l2=300*150+150,l3=150*75+75
    def mlp(X,Y,n_maxouts=3):
        # construct learnable parameters within local scope
      with tf.device('/gpu:0'):
        w1=tf.get_variable("w1", [X.get_shape()[1], 50], initializer=tf.random_normal_initializer())
        b1=tf.get_variable("b1", [50], initializer=tf.constant_initializer(0.5))
        w2=tf.get_variable("w2", [50, 30], initializer=tf.random_normal_initializer())
        b2=tf.get_variable("b2", [30], initializer=tf.constant_initializer(0.5))
        w3=tf.get_variable("w3", [30,25], initializer=tf.random_normal_initializer())
        b3=tf.get_variable("b3", [25], initializer=tf.constant_initializer(0.5))
        w4=tf.get_variable("w4", [25,Y.get_shape()[1]], initializer=tf.random_normal_initializer())
        b4=tf.get_variable("b4", [Y.get_shape()[1]], initializer=tf.constant_initializer(0.0))
        # nn operators
        fc1=tf.nn.relu(tf.matmul(X,w1)+b1)
        # fc1= tf.nn.dropout(fc1, keep_prob=0.5)
        fc2=tf.nn.relu(tf.matmul(fc1,w2)+b2)
        # fc2= tf.nn.dropout(fc2, keep_prob=0.5)
        fc3=tf.nn.relu(tf.matmul(fc2,w3)+b3)
        mo_list=[]
        if n_maxouts>0 :
            w = tf.get_variable('mo_w_0', [25,Y.get_shape()[1]],initializer=tf.random_normal_initializer())
            b = tf.get_variable('mo_b_0', [Y.get_shape()[1]],initializer=tf.constant_initializer(0.0))
            fc4 = tf.matmul(fc3, w) + b
            mo_list.append(w)
            mo_list.append(b)
            for i in range(n_maxouts):
                if i>0:
                    w = tf.get_variable('mo_w_%d' % i, [25,Y.get_shape()[1]],initializer=tf.random_normal_initializer())
                    b = tf.get_variable('mo_b_%d' % i, [Y.get_shape()[1]],initializer=tf.constant_initializer(0.0))
                    mo_list.append(w)
                    mo_list.append(b)
                    fc4=tf.stack([fc4,tf.matmul(fc3, w) + b],axis=-1)
                    fc4 = tf.reduce_max(fc4,axis=-1)
        else:
            fc4=tf.matmul(fc3,w4)+b4
        return fc4

    output=mlp_cnn(xs,keep_prob)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=ys))

    #计算预测数据与实际数据的差异,进行精度检测
    predict = tf.reshape(output, [-1, 2, d_l])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(ys, [-1, 2, d_l]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

    # important step 对所有变量进行初始化
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # with tf.train.MonitoredTrainingSession(
    #         # hooks=[tf.train.StopAtStepHook(last_step=2000),
    #         #        tf.train.NanTensorHook(loss)],
    #         config=config
    #    ) as sess:
    #      sess.run(init)
    #      step=0
    #      while not sess.should_stop():
    #              sess.run(loss)
    #              if step % 50 == 0:
    #                  # saver.save(sess, "tmp_NLP.model", global_step=step)
    #                  # to see the step improvement
    #                  # print(sess.run(accuracy, feed_dict={xs: x_data, ys: y_data}))
    #                  print(step,sess.run(accuracy))
    #              step=step+1
    with tf.device('/gpu:1'):
        with tf.Session(config=config) as sess:
            sess.run(iterator.initializer)
            for step in range(100000):
                sess.run(init)
                # 迭代 1000 次学习，sess.run optimizer
                x_data, y_data = sess.run(next_element)
                # print("x_data:=",x_data)
                # print("y_data:=",y_data)
                #sess.run(iterator.initializer)
                # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
                try:
                    #sess.run(loss)
                    sess.run(train_op, feed_dict={xs: x_data, ys: y_data,keep_prob:0.5})
                    if step % 100 == 0:
                        saver.save(sess, "/temp/tmp_NLP.model", global_step=step)
                        # to see the step improvement
                        print(sess.run(accuracy, feed_dict={xs: x_data, ys: y_data,keep_prob:0.5}))
                        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                        # print(step,sess.run(accuracy))
                except tf.errors.OutOfRangeError:
                       print("chu")
                       break
        sess.close()

