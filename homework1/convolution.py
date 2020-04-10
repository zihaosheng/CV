import numpy as np
import scipy.signal
x = np.array([1,2,1])
h = np.array([0,1,2,3,3,3,1,3,6])
print(scipy.signal.convolve(h, x))#一维卷积运算
print(scipy.signal.convolve(x, h))#一维卷积运算

print(len(scipy.signal.convolve(h, x)))
print(len(h))