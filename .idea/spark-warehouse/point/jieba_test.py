import jieba
import re

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

line="济南生产运营中心庄子_风机10_11#DP故障信息(luoe)[ddd]"
matchObj = re.search(r'\d{1,10}_\d{1,10}|\(.*\)', line, re.M|re.I)
if matchObj:
    print("search --> matchObj.group() : ", matchObj.group())
else:
    print("No match!!")

matchObj= re.search( r'\(.*\)', line, re.M|re.I)
if matchObj:
    print("search --> matchObj.group() : ", matchObj.group())
else:
    print("No match!!")

matchObj= re.search(r'\[.*\]', line, re.M|re.I)
if matchObj:
    print("search --> matchObj.group() : ", matchObj.group())
else:
    print("No match!!")
exit()

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
while(i < row_num):

    if not str(dict_data[i]["name"]).__eq__(""):
       print(str(dict_data[i]["z"]).replace("(","").replace(")","")
              .replace("[","").replace("]","").replace("℃","")
              ,":=",dict_data[i]["name"]
            )

       temp=str(dict_data[i]["z"]).replace("(","").replace(")","")\
            .replace("[","").replace("]","").replace("℃","").replace("）","").replace("（","")\
            .replace("L1","").replace("L2","").replace("L3","").replace("I1","").replace("I2","") \
            .replace("I2","")

       seg_list = jieba.cut(temp)  # 默认是精确模式
       print(", ".join(seg_list))
       print("aaa")
       seg_list = jieba.cut(temp)
       cc=",".join(seg_list).replace("#","").replace("_","")\
           .split(",")
       print(cc)
       if i==0:
          exit()
       print("----------------------------------------------------")
       j=j+1
    i += 1
print("row_num:=",row_num)
print("row_j:=",j)

import tensorflow as tf
