
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


file1 = open("MyFile.txt", "w+")
for el in f:
    file1.write(str(el))

for el in w:
    file1.write(str(el))

from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import numpy as np
sample_rateasd, audio = wavfile.read("./jeden_16bit.wav")
# freqs = np.fft.fftfreq(audio)
# FFT = np.fft.

from scipy import fftpack

# X = fftpack.fft(audio, 160)
# freqs = fftpack.fftfreq(len(audio)) * sample_rateasd

# audio = np.mean(audio, axis=1)
# audio = np.transpose(audio)
from main import *
ffta = np.fft.fft(audio)
plt.figure(3)
plt.subplot(2, 1, 1)
plt.title("FFT of unfiltered vs filtered signal")
FFT = fft.fft(audio)
FFT = FFT[:len(FFT) // 2]
plt.plot(FFT)
plt.subplot(2, 1, 2)
FFT_ = fft.fft(high_pass_filter(audio))
FFT_ = FFT_[:len(FFT_) // 2]
plt.plot(FFT_)
plt.show()
