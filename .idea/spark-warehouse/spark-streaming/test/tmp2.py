
import  tensorflow as tf
import numpy as np
y=np.array([1,3,4])
pi=3.1415926
x=2
h=1
print(np.mean(np.exp(-np.power(x-y,2)/(np.power(h,2)*2.0))/(np.power(pi*2.0,0.5)*h)))
#tf.reduce_sum(tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.linspace(1,1,n))))
