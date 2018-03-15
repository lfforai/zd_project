import tensorflow as tf
print("the model using the version of tensorflow is",tf.__version__)
print("author is: LuoFeng")

#device and serise
#device_code
device_code=['GU','GS','GT','GA','GW','HJ','GL','GR','FJ']#9
device_name={'GU':["逆变器","逆变","整流","整流柜"],
            'GS':["断路器","隔离开关","接地刀闸"],
            'GT':["变压器","电抗器","电抗","电压互感器","电压互感","PT","pt","电流互感器","电流互感","CT","ct"],
            'GA':["电缆","母线","电网","出线"],
            'GW':["箱式供电设备","箱式","供电","机柜电源设备","电源","SVC","SVG"],
            'HJ':["环境监测","环境","监测","天气"],
            'GL':["功率预测","预测","平均"],
            'GR':["汇流箱","汇流柜","直流配电柜","配电柜","直流配电","配电","直流"],
            'FJ':["FJ","风机","风轮机","机舱","主控系统","液压系统","冷却系统","冷却","主控","液压",
                  "风轮机","发电机","转子","叶轮","叶片","轮毂","传动","变速","主轴","主轴承"
                  "齿轮箱","机械刹车","刹车","直驱","双馈","定子","轴承","滑环","变桨","控制箱","控制柜"
                  ,"控制","风速仪","风向标","偏航","测风","电机","计数器","润滑","油泵","油箱","液压","压力","油位",
                  "阀门","过滤器","蓄能器","蓄能","防爆膜","电气控制","变流柜","塔基","机塔","机舱罩","起吊装置","防雷","机架"
                  ,"辅助设备"]}
d_l=device_code.__len__()

#state_code
state_code=['CD','PW','PR','PP','PJ','JR','CN','CU','PF','PV' \
    ,'PA','UR','TM','MS','SS','BF','MA','ST','TS','SA','AD' \
    ,'CA','CT','CP','XD','EF','RR','WP','YD','TP','TR' \
    ,'AN','EN','RA','EX','ZT','ZJ','VA','ZK','ZD','YC','BS','BJ']#43

state_name={'CD':["密度"],'PW':["有功"],'PR':["无功"],
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
            'MS':["运行状态","运行","合闸"],
            'SS':["停止状态","停止","分闸","脱扣"],
            'BF':["中断"],
            'MA':["维护","运维"],
            'ST':["待机","休眠"],'TS':["故障"],'SA':["保护"],'AD':["调节"]
    ,'CA':["报警","警报"],'CT':["温度"],'CP':["压力"],'XD':["湿度"],
    'EF':["效率"],'RR':["辐照度","辐照量","辐照"],
    'WP':["风功率"],'YD':["角度","偏航度","偏航"],'TP':["档位","位置"],
    'TR':["转矩","扭矩","力矩"],
    'AN':["分析仪","测量仪","分析","测量"],
    'EN':["蒸发"],'RA':["雨量","雨"],
    'EX':["极值","极大","极小"],'ZT':["状态","一点多用","多用","一点"],
    'ZJ':["自检"],'VA':["视在功率"],'ZK':["阻抗"],
    'ZD':["震动","振动"],'YC':["远程"],'BS':["闭锁"],'BJ':["变桨"]} #43
s_l=state_code.__len__()

#address_code


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
    seg_list = jieba.cut_for_search(temp)

    #replace
    cc=list(",".join(seg_list).replace("#","").replace("_","").replace(".","") \
                .replace("生产","").replace("运营","").replace("中心","")
                .split(","))
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

print("济南生产运营中心庄子_风机10_10#DP故障信息")
print(text2jieba("济南生产运营中心庄子_风机10_10#DP故障信息"))