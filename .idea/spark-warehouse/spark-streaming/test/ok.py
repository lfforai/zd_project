import tensorflow as tf
import numpy as numpy
a=numpy.array([[1,2,3,4],[4,5,6,9]])
print(a[:,[1,-1]])
K=4
statis=[ [i,dict()] for i in range(K)]#每一组的统计信息记录
c='a'

statis[1][1]["a"]=1
statis[1][1]["a"]=statis[1][1][str(c)]+2
statis[1][1]["b"]=1
c='b'
statis[1][1]["b"]=statis[1][1][str(c)]+2
if not statis[1][1].keys().__contains__("a"):
      print("ok")
print(statis[1][1].__len__())
print(statis[1][1][1])
print(statis)