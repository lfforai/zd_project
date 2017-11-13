
import  tensorflow as tf
import numpy as np
y=np.array([16,36,46,56,56,56,65])
pi=3.1415926
x=54
h=1
print(np.mean(np.exp(-np.power(x-y,2)/(np.power(h,2)*2.0))/(np.power(pi*2.0,0.5)*h)))
#tf.reduce_sum(tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.linspace(1,1,n))))
print(np.linspace(1,3,3))