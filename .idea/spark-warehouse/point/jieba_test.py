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
state_code=['CD','PW','PR','PP','PJ','JR','CN','CU','PF','PV'\
            ,'PA','UR','TM','MS','SS','BF','MA','ST','TS','SA','AD'\
            ,'CA','CT','CP','XD','EF','RR','WP','YD','TP','TR'\
            ,'AN','EN','RA','EX','ZT','ZJ','VA','ZK','ZD','YC','BS','BJ']
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
        print("search --> matchObj.group() : ", matchObj.group())
        temp=temp.replace(matchObj.group(),"")
    else:
        pass

    seg_list = jieba.cut(temp)
    cc=",".join(seg_list).replace("#","").replace("_","") \
        .split(",")
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

line="济南生产运营中心庄子_风机10_11#DP故障信息(luoe)[ddd]"

bid_info = csv.DictReader(open('/media/root/4e73770f-a0a4-492c-b90b-4c93dccfaec32/lf/PointData_201801051031.csv','r'))
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
       print(str(dict_data[i]["z"]),":=",dict_data[i]["name"])
       cc=text2jieba(dict_data[i]["z"])
       print(cc)
       for e in cc:
            if  total_space.get(e):
                total_space[e]=total_space[e]+1
            else:
                total_space[e]=1
       print("----------------------------------------------------")
       j=j+1
    i += 1
print("row_num:=",row_num)
print("row_j:=",j)
total_space["_"]=11
b=zip(total_space.keys(),total_space.values())   #拉成Tuple对组成的List
total_space=list(sorted(b, key=lambda item:item[1]))
total_space=dict(filter(lambda x: True if int(x[1])>10 else False,total_space))
std_one_hot=total_space.keys()#标准化词汇表
print(std_one_hot)

#text分解到one_hot向量,len返回的固定向量长度,10
def text2onehot(text,marking,std_one_hot,device_code=device_code,state_code=state_code,len=20):
    device_code_len=list(device_code).__len__()
    state_code_len=list(state_code).__len__()
    result_mark=list(np.zeros(device_code_len+state_code_len))

    text_list=list(text2jieba(text))#分解为
    # 景峡一期每日发电量 := G_XJJX_HM_01W_001GL_PJ001
    # ['景峡', '一期', '每日', '发电量']
    result=[int(list(std_one_hot).index("_"))]*len #初始化·["_","_","_"]
    i=0
    for e in text_list:
        if list(std_one_hot).__contains__(e):
           index_N=int(list(std_one_hot).index(e))
           result[i]=index_N
           i=i+1
        else:
           pass
    marking_list=str(marking)[-8:-3].split("_")
    result_mark[device_code.index(str(marking_list[0]))]=1
    result_mark[state_code.index(str(marking_list[1]))+device_code_len]=1
    return [result,result_mark]#one_hot向量和所属标记

print(text2onehot("红星二场一期光伏日发电量"," G_XJTX_HM_01P_001GL_PJ001",std_one_hot))

#储存为onehot-mark的格式