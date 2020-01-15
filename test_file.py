
import numpy as np
from scipy import fft as fft

asd = np.array([1, 2, 3])

qwe = np.array([4, 5, 6])

zxc = np.array([[3,7,5], [8,4,3], [2,4,9]])
aaa = np.array([[1,2,3], [1,2,5], [4,5,7]])
# print(zxc)

zxc = np.matrix(zxc)
aaa = np.matrix(aaa)

all = list()
all.append(zxc)
all.append(aaa)
# print(len(all))

# avg = int
# sum = int

#
# avg = sum/len(all)
# print(avg)
sum = 0
for i in range(0, len(all)):
    # if not i+1 >=len(all):
    sum += (np.array(all[i]))

avg = sum / len(all)

# print(avg)


a= [1, 2, 3, 4]
import scipy.fftpack as s
a = s.fft(a)
# a = a[:len(a) // 2]
# print(a)

x = [[1,2], [2,3]]
y = fft(x)
print(y)

q = [1,2]
w = [3,4]
print(np.multiply(q,w))

f = [1,2,3,4,5,3,5,3,5,3,4,3,4];
print(np.sum(f[1:3]))
