import tensorflow as tf
import csv
import minpy.numpy as np
import minpy.numpy.random as random
import time

print("the model using the version of tensorflow is",tf.__version__)
print("author is: LuoFeng")

#零、device and serise-----------------------------------------------------------
#device_code
device_code=['FJ','GU','GS','GT','GA','GW','HJ','GL','GR']#9
device_name={'FJ':["FJ","风机","风轮机","机舱","主控系统","液压系统","冷却系统","冷却","主控","液压",
                   "风轮机","发电机","转子","叶轮","叶片","轮毂","传动","变速","主轴","主轴承"
                   "齿轮箱","机械刹车","刹车","直驱","双馈","定子","轴承","滑环","变桨","控制箱","控制柜"
                  ,"控制","风速仪","风向标","偏航","测风","电机","计数器","润滑","油泵","油箱","液压","压力","油位",
                    "阀门","过滤器","蓄能器","蓄能","防爆膜","电气控制","变流柜","塔基","机塔","机舱罩","起吊装置","防雷","机架"
                    ,"辅助设备"],
            'GU':["逆变器","逆变","整流","整流柜"],
            'GS':["断路器","断路","隔离开关","接地","隔离","接地刀闸"],
            'GT':["变压器","电抗器","电抗","电压互感器","电压互感","PT","pt","电流互感器","电流互感","CT","ct"],
            'GA':["电缆","母线","电网","出线"],
            'GW':["箱式供电设备","箱式","供电","机柜电源设备","电源","SVC","SVG"],
            'HJ':["环境监测","环境","监测","天气"],
            'GL':["功率预测","预测","平均"],
            'GR':["汇流箱","汇流柜","直流配电柜","配电柜","直流配电","配电","直流"]
           }
d_l=device_code.__len__()

#state_code
state_code=['ZT','TS','CD','PW','PR','PP','PJ','JR','CN','CU','PF','PV' \
    ,'PA','UR','TM','MS','SS','BF','MA','ST','SA','AD' \
    ,'CA','CT','CP','XD','EF','RR','WP','YD','TP','TR' \
    ,'AN','EN','RA','EX','ZJ','VA','ZK','ZD','YC','BS','BJ']#43

state_name={'MS':["运行状态","运行","合闸"],
            'ZT':["状态","一点多用","多用","一点"],
            'TS':["故障"],
            'CD':["密度"],'PW':["有功功率"],'PR':["无功功率"],
            'PP':["功率因数"],
            'PJ':["有功电量"],
            'JR':["无功电量"],
            'CN':["转速"],
            'CU':["速度","风速","风向","风向角"],
            'PF':["频率"],
            'PV':["电压"],
            'PA':["电流"],
            'UR':["利用率"],
            'TM':["时间","time","小时"],
            'SS':["停止状态","停止","停机","分闸","脱扣"],
            'BF':["中断"],
            'MA':["维护","运维"],
            'ST':["待机","休眠"],'SA':["保护"],'AD':["调节"]
    ,'CA':["报警","警报"],'CT':["温度"],'CP':["压力"],'XD':["湿度"],
    'EF':["效率"],'RR':["辐照度","辐照量","辐照"],
    'WP':["风功率"],'YD':["角度","偏航度"],'TP':["档位","位置"],
    'TR':["转矩","扭矩","力矩"],
    'AN':["分析仪","测量仪","分析","测量","检测"],
    'EN':["蒸发"],'RA':["雨量","雨"],
    'EX':["极值","极大","极小"],
    'ZJ':["自检"],'VA':["视在功率"],'ZK':["阻抗"],
    'ZD':["震动","振动"],'YC':["远程"],'BS':["闭锁"],'BJ':["变桨"]} #43
s_l=state_code.__len__()
# temp_x=['风机', '11', '故障', '停机', '等级']
# for s in state_name:
#     break_num=0
#     for s1 in state_name[s]:
#         if temp_x.__contains__(s1):
#            temp_p_s_rules=s
#            bayes_use_d=0
#            break_num=1
#            break
#     if break_num==1:
#         break
# print(temp_p_s_rules)
# exit()

#address_code
address_code=[]#read from csv
def address_read(filename='/address_code.csv'):
    print("start building address code.")
    rezult=[]
    bid_info = csv.DictReader(open(filename,'r'))
    dict_data = []
    for lines in bid_info:
        if bid_info.line_num == 1:
            continue
        else:
            dict_data.append(lines)
    row_num = len(dict_data)
    #print('this is all the data---' + str(dict))
    for i in range(row_num):
        if not str(dict_data[i]["address"]).__eq__(""):
           rezult.append(dict_data[i]["address"])
    return rezult
address_code=address_read()

#define frequence dist
device_code_space=dict()
state_code_space=dict()
device_pro=dict()
state_pro=dict()

#----------------------------------------------------------------------
#一、data preprocessing
print("Starting Data Preprocessing")

#（一）jieba
import re
import jieba

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
    seg_list = jieba.cut(temp)
    #.cut_for_search(temp)

    #replace
    cc=list(",".join(seg_list).replace("#","").replace("_","").replace(".","") \
                .replace("生产","").replace("运营","").replace("中心","")
                .split(","))
    cc=filter(lambda x:not address_code.__contains__(x),cc)
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
    print("be starting to create lexicon!,wait for monent")
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
        #print(sum_temp)
        if sum_temp>0:
            for d,x in x_n.items():
                device_code_space[d_n][d]=float(x)/sum_temp
    print("-----------------------")
    for d_n,x_n in state_code_space.items():
        sum_temp=sum(list(dict(x_n).values()))
        #print(sum_temp)
        if sum_temp>0:
            for d,x in x_n.items():
                state_code_space[d_n][d]=float(x)/sum_temp

    #print(device_pro)
    sum_temp=sum(list(dict(device_pro).values()))
    #print(sum_temp)
    for d_n in device_pro:
        device_pro[d_n]=float(device_pro[d_n])/sum_temp

    #print(state_pro)
    sum_temp=sum(list(dict(state_pro).values()))
    for d_n in state_pro:
        state_pro[d_n]=float(state_pro[d_n])/sum_temp
    return device_code_space,state_code_space,device_pro,state_pro

def forecast_Bayes(filename,device_code_space,state_code_space,device_pro,state_pro,model=1):
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
    #model=0(use bayes),1(use  rules engine and bayes together)
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
                if device_code_space[e].get(e1)==None:
                    temp_sum_mid=temp_sum_mid*0.00000001
                else:
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
                if state_code_space[e].get(e1)==None:
                    temp_sum_mid=temp_sum_mid*0.00000001
                else:
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
    j=0
    for i in range(row_num):
        if not str(dict_data[i]["name"]).__eq__(""):
            temp_x=text2jieba(str(dict_data[i]["z"]))
            temp_y=str(dict_data[i]["name"]).replace(" ","")[-8:-3].split("_")

            #print(temp_x)
            if model==0:# only use bayes
               temp_p_d,temp_p_s=Bayes(temp_x)
            else:#use bayes and rules engine
               if model==1:
                   # print("model==",1)
                   bayes_use_d=1#device_code
                   bayes_use_s=1#state_code
                   #use rules engine
                   for d in device_name:
                       break_num=0
                       index_min=device_name[d].__len__()
                       for d1 in device_name[d]:
                          if temp_x.__contains__(d1):
                             index_temp=device_name[d].index(d1)
                             if index_temp<index_min:
                                index_min=index_temp
                                temp_p_d_rules=d
                                bayes_use_d=0
                                break_num=1
                       if break_num==1:
                          break

                   index_max=-1
                   for s in state_name:
                       for s1 in state_name[s]:
                          if temp_x.__contains__(s1):
                             index_temp=temp_x.index(s1)
                             if index_temp>index_max:
                                index_max=index_temp
                                temp_p_s_rules=s
                                bayes_use_s=0
                       # if break_num==1:
                       #    break
                   #use bayes
                   if bayes_use_s==0 and bayes_use_d==1:#device use rules,state use
                      temp_p_d,temp_p_s=Bayes(temp_x)
                      temp_p_s=temp_p_s_rules
                   else:
                       if bayes_use_s==1 and bayes_use_d==0:
                          temp_p_d,temp_p_s=Bayes(temp_x)
                          temp_p_d=temp_p_d_rules
                       else:
                          if bayes_use_s==1 and bayes_use_d==1:
                             temp_p_d,temp_p_s=Bayes(temp_x)
                          else:#bayes_use_s==0 and bayes_use_d==0
                             temp_p_d=temp_p_d_rules
                             temp_p_s=temp_p_s_rules

            #specail deal
            if temp_x.__contains__("运行") and temp_x.__contains__("状态")\
               and abs(temp_x.index("运行")-temp_x.index("状态"))==1:
               temp_p_s="MS"

            if temp_x.__contains__("状态") and temp_x.__contains__("显示") \
                    and abs(temp_x.index("状态")-temp_x.index("显示"))==1:
                temp_p_s="MS"

            if temp_x.__contains__("停机") and temp_x.__contains__("状态") \
                    and abs(temp_x.index("停机")-temp_x.index("状态"))==1:
                temp_p_s="SS"

            if temp_x.__contains__("有功") and (temp_x.__contains__("电量")) \
                    and abs(temp_x.index("有功")-temp_x.index("电量"))==1:
                temp_p_s="PJ"

            if temp_x.__contains__("有功") and (temp_x.__contains__("发电量")) \
                    and abs(temp_x.index("有功")-temp_x.index("发电量"))==1:
                temp_p_s="PJ"

            if temp_x.__contains__("无功") and (temp_x.__contains__("电量")) \
                        and abs(temp_x.index("无功")-temp_x.index("电量"))==1:
                temp_p_s="JR"


            if temp_x.__contains__("无功") and (temp_x.__contains__("发电量")) \
                    and abs(temp_x.index("无功")-temp_x.index("发电量"))==1:
                temp_p_s="JR"

            if temp_x.__contains__("有功") and (temp_x.__contains__("功率")) \
                    and abs(temp_x.index("有功")-temp_x.index("功率"))==1:
                temp_p_s="PW"

            if temp_x.__contains__("无功") and (temp_x.__contains__("功率")) \
                    and abs(temp_x.index("无功")-temp_x.index("功率"))==1:
                temp_p_s="PR"

            if str(temp_p_d).__eq__(temp_y[0]):
                correct_device_num=correct_device_num+1
                # print("device right!")
            else:
                pass
                print("ok:=",temp_x)
                print(dict_data[i]["name"])
                print(str(dict_data[i]["z"]))
                print("device 错误！")
                print("device:",temp_p_d)
                print("device:=",temp_x)
                print("device:=",temp_y[0])
                if bayes_use_d==0:
                   use="not be used!"
                else:
                   use="be used!"
                print("bayes_use_d:",use)
                print("----------------------------------")
            if str(temp_p_s).__eq__(temp_y[1]):
                correct_state_num=correct_state_num+1
                # print("state right!")
            else:
                pass
                print("ok:=",temp_x)
                print(dict_data[i]["name"])
                print(str(dict_data[i]["z"]))
                print("state 错误！")
                print("state:",temp_p_s)
                print("state:=",temp_x)
                print("state:=",temp_y[1])
                if bayes_use_d==0:
                    use="not be used!"
                else:
                    use="be used!"
                print("bayes_use_s:",use)
                print("----------------------------------")
            j=j+1
    return  correct_device_num/j,correct_state_num/j
device_code_space,state_code_space,device_pro,state_pro=read2words()


print("be starting to expand lexicon of device_name and state_name!")
for key_o in device_code_space:
    for d_o in device_code_space[key_o]:
        oneself_mask=True #suppose that the key_o is monopolize at  beginning.
        for key_e in device_code_space:
            if not str(key_o).__eq__(str(key_e)):
                 if list(device_code_space[key_e]).__contains__(d_o):
                    oneself_mask=False
                    break
        if oneself_mask==True:
           if not device_name[key_o].__contains__(d_o):
               device_name[key_o].append(d_o)
               #print("increas device_name",key_o)

for key_o in state_code_space:
    for d_o in state_code_space[key_o]:
        oneself_mask=True #suppose that the key_o is monopolize at  beginning.
        for key_e in state_code_space:
            if not str(key_o).__eq__(str(key_e)):
                if list(state_code_space[key_e]).__contains__(d_o):
                    oneself_mask=False
                    break
        if oneself_mask==True:
            if not state_name[key_o].__contains__(d_o):
                state_name[key_o].append(d_o)
                #print("increas state_name",key_o)
exit()
print(device_name)
# print(text2jieba("1#站用变]一次有功功率"))
# print(state_code_space["CT"])
# exit()
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
