import jieba
import re
import numpy as np

#
# s="济南生产运营中心庄子_风机10_变桨1故障字2"
# seg_list = jieba.cut(s, cut_all=True)
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式
#
# seg_list = jieba.cut(s, cut_all=False)
# print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
#
# seg_list = jieba.cut(s)  # 默认是精确模式
# print(", ".join(seg_list))
#
# seg_list = jieba.cut_for_search(s)  # 搜索引擎模式
# print(", ".join(seg_list))
#

import csv
device_code=['GU','GS','GT','GA','GW','HJ','GL','GR','FJ']
d_l=device_code.__len__()

state_code=['CD','PW','PR','PP','PJ','JR','CN','CU','PF','PV'\
            ,'PA','UR','TM','MS','SS','BF','MA','ST','TS','SA','AD'\
            ,'CA','CT','CP','XD','EF','RR','WP','YD','TP','TR'\
            ,'AN','EN','RA','EX','ZT','ZJ','VA','ZK','ZD','YC','BS','BJ']
s_l=state_code.__len__()

for i in range(s_l-d_l):
    device_code.append("wwww")
d_l=device_code.__len__()
# matchObj=re.search(r'\.\d+', "adsfasc23", re.M|re.I)
# if matchObj:
#    print(matchObj.group())
# exit()
is_digt=False
def text2jieba(temp):
    temp=str(temp) \
        .replace("℃","") \
        .replace("L1","").replace("L2","").replace("L3","").replace("I1","").replace("I2","") \
        .replace("I2","").replace("#","").replace(" ","").replace("]","") \
        .replace("[","").replace("(","").replace(")","").replace("（","") \
        .replace("）","")
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

    seg_list = jieba.cut(temp)
    cc=list(set(",".join(seg_list).replace("#","").replace("_","") \
        .split(",")))
    def mapfunc(x):
        global  is_digt
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

#line="济南生产运营中心庄子_风机10_11#DP故障信息(luoe)[ddd]"
def read2words(filename='/media/root/4e73770f-a0a4-492c-b90b-4c93dccfaec32/lf/PointData_201801051031.csv'):
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
           # print(str(dict_data[i]["z"]),":=",dict_data[i]["name"])
           cc=text2jieba(dict_data[i]["z"])
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
    total_space=dict(filter(lambda x: True if int(x[1])>0 else False,total_space))
    std_one_hot=total_space.keys()#标准化词汇表
    # print(std_one_hot)
    print("word词库已经建立完成")
    return std_one_hot

#text分解到one_hot向量,len返回的固定向量长度,10
def text2onehot(text,marking,std_one_hot=[],device_code=device_code,state_code=state_code,len=20):
    device_code_len=d_l
    state_code_len=s_l
    result_mark=list(np.zeros(device_code_len+state_code_len))

    text_list=list(text2jieba(text))#分解为
    if text_list.__len__()>len:
       text_list=text_list[-len+1:-1]#如果超过len截断前面的数据只保留后面
    # 景峡一期每日发电量 := G_XJJX_HM_01W_001GL_PJ001
    # ['景峡', '一期', '每日', '发电量']
    result=[int(list(std_one_hot).index("_"))]*len #初始化·["_","_","_"]
    i=0
    for e in text_list:
        if std_one_hot.__contains__(e):
           index_N=int(list(std_one_hot).index(e))
           result[i]=index_N
           i=i+1
        else:
           pass
    result=sorted(result)
    marking_list=str(marking).replace(" ","")[-8:-3].split("_")
    if device_code.__contains__(str(marking_list[0])) and state_code.__contains__(str(marking_list[1])) :
       result_mark[device_code.index(str(marking_list[0]))]=1
       result_mark[state_code.index(str(marking_list[1]))+device_code_len]=1
       return [result,result_mark]#one_hot向量和所属标记
    else:
       return ['bad','bad']

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
def text2x_y(filename='/media/root/4e73770f-a0a4-492c-b90b-4c93dccfaec32/lf/PointData_201801051031.csv',std_one_hot=[]):
    print("开始生成样本x_vs_y!")
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
            x,y=text2onehot(str(dict_data[i]["z"]),str(dict_data[i]["name"]),std_one_hot)
            result.append([x,y])
            # print(x,y)
            # print("----------------------------------------------------")
            j=j+1
        i += 1
    print("结束生成样本x_vs_y!")
    return result

#创建词库
std_one_hot=read2words(filename='/media/root/4e73770f-a0a4-492c-b90b-4c93dccfaec32/lf/PointData_201801051031.csv')

# x,y=text2onehot("红星二场一期光伏日发电量"," G_XJTX_HM_01P_001GL_PJ001",std_one_hot)
# z=onehot2mark(result_mark=y)
# print(x)
# print(y)
# print(z)
#一、开始执行
#（1）创建样本储存为onehot-mark的格式
result=text2x_y(std_one_hot=std_one_hot)
result=list(filter(lambda x:True if not str(x[0]).__eq__("bad") else False,result))
# print(result)

#（2）tensoflow神经网
import tensorflow as tf

#相关参数设定
x_len=list(result[0][0]).__len__()
print("x_len:=",x_len)
y_len=list(result[0][1]).__len__()
print("y_len:=",y_len)
batchSize=128
img=tf.convert_to_tensor(np.array([e[0] for e in result]))
label=tf.convert_to_tensor(np.array([e[1] for e in result]))

#从python的np.array中获获取分批生成
def batch_input(img,label, batchSize):
    min_after_dequeue = 1000
    capacity = min_after_dequeue+3*batchSize
    # 预取图像和label并随机打乱，组成batch，此时tensor rank发生了变化，多了一个batch大小的维度
    exampleBatch,labelBatch = tf.train.shuffle_batch([img, label],batch_size=batchSize, capacity=capacity,
                                                     min_after_dequeue=min_after_dequeue)
    return exampleBatch,labelBatch

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, x_len])
ys = tf.placeholder(tf.float32, [None, y_len])

with tf.variable_scope("G", reuse=tf.AUTO_REUSE):
    w1=tf.get_variable("w0", [20, 300])
    b1=tf.get_variable("b0", [300])
    w2=tf.get_variable("w1", [300, 150])
    b2=tf.get_variable("b1", [150])
    w3=tf.get_variable("w2", [150, 75])
    b3=tf.get_variable("b2", [86])

    #全链接神经网，l1=input×300+300, l2=300*150+150,l3=150*75+75
    def mlp(X,Y,n_maxouts=5):
        # construct learnable parameters within local scope
        w1=tf.get_variable("w0", [X.get_shape()[1], 300], initializer=tf.random_normal_initializer())
        b1=tf.get_variable("b0", [300], initializer=tf.constant_initializer(0.0))
        w2=tf.get_variable("w1", [300, 150], initializer=tf.random_normal_initializer())
        b2=tf.get_variable("b1", [150], initializer=tf.constant_initializer(0.0))
        w3=tf.get_variable("w2", [150, Y.get_shape()[1]], initializer=tf.random_normal_initializer())
        b3=tf.get_variable("b2", [Y.get_shape()[1]], initializer=tf.constant_initializer(0.0))
        #w4=tf.get_variable("w3", [75,output_dim], initializer=tf.random_normal_initializer())
        #b4=tf.get_variable("b3", [output_dim], initializer=tf.constant_initializer(0.0))
        # nn operators
        fc1=tf.nn.relu(tf.matmul(input,w1)+b1)
        # fc1= tf.nn.dropout(fc1, keep_prob=0.5)
        fc2=tf.nn.relu(tf.matmul(fc1,w2)+b2)
        # fc2= tf.nn.dropout(fc2, keep_prob=0.5)
        fc3=tf.nn.relu(tf.matmul(fc2,w3)+b3)
        # mo_list=[]
        # if n_maxouts>0 :
        #     w = tf.get_variable('mo_w_0', [75,output_dim],initializer=tf.random_normal_initializer())
        #     b = tf.get_variable('mo_b_0', [output_dim],initializer=tf.constant_initializer(0.0))
        #     fc4 = tf.matmul(fc3, w) + b
        #     mo_list.append(w)
        #     mo_list.append(b)
        #     for i in range(n_maxouts):
        #         if i>0:
        #             w = tf.get_variable('mo_w_%d' % i, [75,output_dim],initializer=tf.random_normal_initializer())
        #             b = tf.get_variable('mo_b_%d' % i, [output_dim],initializer=tf.constant_initializer(0.0))
        #             mo_list.append(w)
        #             mo_list.append(b)
        #             fc4=tf.stack([fc4,tf.matmul(fc3, w) + b],axis=-1)
        #             fc4 = tf.reduce_max(fc4,axis=-1)
        # else:
        #     fc4=tf.matmul(fc3,w4)+b4
        return fc3, [w1,b1,w2,b2,w3,b3]

    output=mlp(xs,ys)
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
init = tf.initialize_all_variables()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)
    # 迭代 1000 次学习，sess.run optimizer
    for step in range(2000):
        x_data,y_data=batch_input(img,label,batchSize)
        # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
        sess.run(train_op, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            saver.save(sess, "tmp_NLP.model", global_step=step)
            # to see the step improvement
            print(sess.run(accuracy, feed_dict={xs: x_data, ys: y_data}))
sess.close()

