import re
import numpy as np
import jieba

# import numpy as np
# import numpy.random as random
# import time
# x= random.rand(1024, 1024)
# y= random.rand(1024, 1024)
# st = time.time()
# for i in range(10000):
#     z= np.dot(x, y)
# print('time: {:.3f}.'.format(time.time()-st))

import minpy.numpy as np
import minpy.numpy.random as random
import time
# x= random.rand(1024, 1024)
# y= random.rand(1024, 1024)
# st = time.time()
# for i in range(10000):
#     z= np.dot(x, y)
# z.asnumpy()
# print('time: {:.3f}.'.format(time.time()-st))
# exit()
#
# s="金紫山一期25号风机3#变桨电容柜体温度(℃)"
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
# #
# # 沈阳驼山一期风机22.1#变桨速率 G_DBTS_SY_01W_022FJ_CN004
# # [0, 0, 7787, 7907, 7918, 7932]

import csv
device_code=['GU','GS','GT','GA','GW','HJ','GL','GR','FJ']#9
d_l=device_code.__len__()

state_code=['CD','PW','PR','PP','PJ','JR','CN','CU','PF','PV'\
            ,'PA','UR','TM','MS','SS','BF','MA','ST','TS','SA','AD'\
            ,'CA','CT','CP','XD','EF','RR','WP','YD','TP','TR'\
            ,'AN','EN','RA','EX','ZT','ZJ','VA','ZK','ZD','YC','BS','BJ']#43
s_l=state_code.__len__()
#
# for i in range(s_l-d_l):
#     device_code.append("wwww")
# d_l=device_code.__len__()
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
    cc=list(set(",".join(seg_list).replace("#","").replace("_","").replace(".","") \
        .split(",")))

    # def mapfunc(x,is_digt=is_digt):
    #     if str(x).__eq__('.'):
    #         return ""
    #     if str(x).isdigit() and is_digt==False:
    #         is_digt=True
    #         return x
    #     else:
    #         if str(x).isdigit() and is_digt==True:
    #             return ""
    #         else:
    #             return x
    #
    # cc=list(map(mapfunc,cc))
    cc=list(filter(lambda x:str(x)!="",cc))
    return cc

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
    j=0

    #设备的词频率矩阵
    device_code_space=dict()
    for e in device_code:
        device_code_space[str(e)]={"_":0}

    #流水码的词频矩阵
    print("----------------------------")
    state_code_space=dict()
    for e in state_code:
        state_code_space[str(e)]={"_":0}

    #设备词频率['GU'=0.9,'GS'=0.1,'GT'=0,'GA'=0,'GW'=0,'HJ'=0,'GL'=0,'GR'=0,'FJ'=0]
    device_pro=dict(zip(device_code,[0.0]*d_l))

    #流水码词频率
    state_pro=dict(zip(state_code,[0.0]*s_l))

    #print('this is all the data---' + str(dict))
    for i in range(row_num):
        if not str(dict_data[i]["name"]).__eq__(""):
          temp_x=text2jieba(str(dict_data[i]["z"]))
          #print(temp_x)
          temp_y=str(dict_data[i]["name"]).replace(" ","")[-8:-3].split("_")

          #统计设备和流水号的词频率
          if not device_pro.get(temp_y[0])==None:
             device_pro[temp_y[0]]=device_pro[temp_y[0]]+1
          if not state_pro.get(temp_y[1])==None:
             state_pro[temp_y[1]]=state_pro[temp_y[1]]+1

          device_dict=device_code_space.get(temp_y[0])#设备码词频率
          state_dict=state_code_space.get(temp_y[1])#流水码词频率

          if device_dict:
             for j in range(temp_x.__len__()):
                if not device_code_space.get(temp_y[0]).get(temp_x[j])==None:
                   device_code_space[temp_y[0]][temp_x[j]]=device_code_space[temp_y[0]][temp_x[j]]+1
                else:
                   #print("+0")
                   device_code_space[temp_y[0]][temp_x[j]]=1

          if state_dict:
             for j in range(temp_x.__len__()):
                  if not state_code_space.get(temp_y[1]).get(str(temp_x[j]))==None:
                     state_code_space[temp_y[1]][temp_x[j]]=state_code_space[temp_y[1]][temp_x[j]]+1
                  else:
                     state_code_space[temp_y[1]][temp_x[j]]=1
          j=j+1

    #将频率转化为概率
    for d_n,x_n in device_code_space.items():
        sum_temp=sum(list(dict(x_n).values()))
        print(sum_temp)
        if sum_temp>0:
            for d,x in x_n.items():
                device_code_space[d_n][d]=float(x)/sum_temp

    print("-----------------------")
    for d_n,x_n in state_code_space.items():
        sum_temp=sum(list(dict(x_n).values()))
        print(sum_temp)
        if sum_temp>0:
            for d,x in x_n.items():
                state_code_space[d_n][d]=float(x)/sum_temp

    print(device_pro)
    sum_temp=sum(list(dict(device_pro).values()))
    print(sum_temp)
    for d_n in device_pro:
        device_pro[d_n]=float(device_pro[d_n])/sum_temp

    print(state_pro)
    sum_temp=sum(list(dict(state_pro).values()))
    for d_n in state_pro:
        state_pro[d_n]=float(state_pro[d_n])/sum_temp
    return device_code_space,state_code_space,device_pro,state_pro

def forecast_Bayes(filename,device_code_space,state_code_space,device_pro,state_pro):
    print("开始进行贝叶斯分类！")
    bid_info = csv.DictReader(open(filename,'r'))
    dict_data = []
    for lines in bid_info:
        if bid_info.line_num == 1:
            continue
        else:
            dict_data.append(lines)
    row_num = len(dict_data)

    correct_device_num=0
    correct_state_num=0

    #贝叶斯概率计算 P（S1|A1,A2）=P（A1|s1）×p（A2|s1）×p（s1）/（sum（P（A1|sn）×p（A2|sn）×p（sn）））
    def Bayes(list_jieba,device_code_space=device_code_space,state_code_space=state_code_space
              ,device_pro=device_pro,state_pro=state_pro):

        #计算出的所有概率的结果
        list_pro_rezult_device=[]
        list_pro_rezult_state=[]

        #计算devcie的结果
        #1\计算sum（P（A1|sn）×p（A2|sn）×p（sn））,假设A1..An相互独立
        sum_device=[]
        for e in device_code_space:
            temp_sum_mid=1
            for e1 in list_jieba:
                temp_sum_mid=temp_sum_mid*device_code_space[e][e1]
            list_pro_rezult_device.append(temp_sum_mid*device_pro[e])

        #1\计算概率
        sum_temp=sum(list_pro_rezult_device)
        list_pro_rezult_device=np.array(list_pro_rezult_device)/sum_temp

        #计算state的结果
        sum_device=[]
        for e in state_code_space:
            temp_sum_mid=1
            for e1 in list_jieba:
                temp_sum_mid=temp_sum_mid*state_code_space[e][e1]
            list_pro_rezult_state.append(temp_sum_mid*state_pro[e])
        #1\计算概率
        sum_temp=sum(list_pro_rezult_state)
        list_pro_rezult_state=np.array(list_pro_rezult_state)/sum_temp

        #计算device中的最大概率
        list_d=list(enumerate(list_pro_rezult_device))
        value_max_index_d=0
        value_max_d=0
        for e in list_d:
          if e[1]>value_max_d:
             value_max_d=e[1]
             value_max_index_d=e[0]
          else:
             pass
        result_d=device_code[value_max_index_d]

        #计算device中的最大概率F
        list_s=list(enumerate(list_pro_rezult_state))
        value_max_index_s=0
        value_max_s=0
        for e in list_s:
            if e[1]>value_max_s:
                value_max_s=e[1]
                value_max_index_s=e[0]
            else:
                pass
        result_s=state_code[value_max_index_s]

        return result_d,result_s

    for i in range(row_num):
        if not str(dict_data[i]["name"]).__eq__(""):
            temp_x=text2jieba(str(dict_data[i]["z"]))
            #print(temp_x)
            temp_y=str(dict_data[i]["name"]).replace(" ","")[-8:-3].split("_")
            temp_p_d,temp_p_s=Bayes(temp_x)
            if str(temp_p_d).__eq__(temp_y[0]):
               correct_device_num=correct_device_num+1
            if str(temp_p_s).__eq__(temp_y[1]):
               correct_state_num_num=correct_state_num+1
    return  correct_device_num/row_num,correct_state_num_num/row_num

# print(text2jieba("1#站用变]一次有功功率"))
device_code_space,state_code_space,device_pro,state_pro=read2words()
# total_space=device_code_space["FJ"]
# b=zip(total_space.keys(),total_space.values())   #拉成Tuple对组成的List
# total_space=list(sorted(b, key=lambda item:item[1]))
#total_space=dict(filter(lambda x: True if int(x[1])>10 or int(x[1])==0 else False,total_space))
# print(total_space)
# print("---------")
# print(device_pro)
# print(state_pro)
print(forecast_Bayes(filename='/PointData_201801051031.csv',device_code_space=device_code_space,state_code_space=state_code_space
                     ,device_pro=device_pro,state_pro=state_pro))