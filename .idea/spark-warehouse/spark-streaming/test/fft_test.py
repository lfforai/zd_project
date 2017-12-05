import numpy as np

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    print(k)
    print(n)
    print(k.shape)
    print(x.shape)
    print(x)
    M = np.exp(-2j * np.pi * k * n / N)
    print(M)
    return np.dot(M, x)


x= np.random.random(5)
# DFT_slow(x)
#
# z=np.asarray([[1,2,3],[4,5,6]])
# b=np.asarray([1,2,3])
#
# print(np.dot(z,b.reshape((3,1))))

# # for i in range (1000000):
# #     x= np.random.random(1000)
# #     np.fft.fft(x)
# #
#
# print("over")
# x= np.random.random(10)
# print(x)
# print(DFT_slow(x))
# w=np.fft.fft(x)
# a=[[e.real,e.imag] for e  in w]
# print(a)
#
# x=2+3j
# x.imag
# y=1-4j
# z=x/y
# print(z.imag)
# print("x:=====",x/y)
#
# N=5
# n = np.arange(N)
# k = n.reshape((N, 1))
# print(k)

# import numpy as np
# from matplotlib.pyplot import plot, show
# x = np.linspace(0, 2 * np.pi, 100) #创建一个包含30个点的余弦波信号
# wave = np.cos(x)
# transformed = np.fft.fft(wave)  #使用fft函数对余弦波信号进行傅里叶变换。
# print(np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9))  #对变换后的结果应用ifft函数，应该可以近似地还原初始信号。
# plot(transformed)  #使用Matplotlib绘制变换后的信号。
# show()
# print(transformed)
# a=np.fft.fft(x)
# print(a)
# print(a[0:3])
a_b=np.sum(np.asarray([1+2j,1+3j,1+4j])[0:2]*[0.6,0.4])
print(a_b)