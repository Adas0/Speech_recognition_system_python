##############################          LINKS           #################################


# https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# https://python-speech-features.readthedocs.io/en/latest/

##########################################################################################

import main
from main import audio, sample_rate
# from python_speech_features import mfcc
import python_speech_features
import numpy
import numpy as np
from statistics import mean


# print(audio.shape)

# frame 20ms calculate
frame_time = 0.02
print(len(audio))
print(len(audio)/(frame_time * sample_rate))

# obliczenie nakładkowania ramek czasowych (25%)
winstep = 0.02

# liczba wspolczynnikow mfcc
numcep = 13

# liczba filtrow (14-20)
filters_amount= 20

# wielkosc ramki FFT
nfft = 512

# najnizsza czestotliwosc filtru w melach, 80 jako najnizsze f0 czlowieka
lowfreq = 80

# najwyzsza czestotliwosc filtru w melach
highfreq = sample_rate/2

# preemfaza - nie wiadomo jaka jednostka w kazdym razie od 0-1
preemph = 0.9

# jakies wygładzenie wspolczynnikow cepstralnych (0-?),
lifter = 14

# wspolczynniki cepstralne - jesli true to zamienia na logarytmiczne o takiej samej energii
log_or_not = True

# okno
window = numpy.hamming




# mfcc calculation
def getMFCCa():
    MFCC = python_speech_features.base.mfcc(audio, samplerate=16000, winlen=0.025, winstep=0.01,
                                            numcep=13, nfilt=20, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                                            ceplifter=22, appendEnergy=True, winfunc=numpy.hamming)
    return MFCC


def function():
    pass


  #297x13 to oznacza wiec mamy 297 ramek (160 ktore nachodza na siebie (25% długości nachodzenia)


# def getDeltas():
#     return python_speech_features.base.delta(MFCC, MFCC.shape[0])


MFCC = getMFCCa()
print(MFCC)
print("mfcc shape", MFCC.shape)
# Deltas = getDeltas()

# print(Deltas.shape)
# print('asd')


#
# a = [1, 2, 3, 5]
# b = [3, 5, 6, 8]
# c = [4, 5, 6, 4]
# d = [1, 4, 7, 8]
# e = a,b,c,d
#
# asd = list()
# # asd.append(a,b)
# # print(numpy.add(a, b))
#
# framed_signal = list()
# print(e)
# first_quater_index = int(0.25 * len(e[0]))
# last_quater_index = int(0.75 * len(e[0]))
# end = len(e[0])
#
# print(np.add(a,b))

# for i in range(0, len(e)):
#     #dla pierwszego elementu
#     if not i+1 > len(e):
#         if i == 0:
#              framed_signal.append(e[i][0:last_quater_index])
#
#              first_part = int
#              first_part = e[i][0: first_quater_index]
#              last_part = int
#              last_part = e[i][last_quater_index : end]
#              print(np.mean(1, 5))
#              print(first_part, last_part)
#              print(np.average(first_part, last_part))
#              # print(asd)
#              framed_signal.append(np.average(e[i][last_quater_index : end], e[i+1][0: first_quater_index]))
#         elif not i == len(e):
#         # framed_signal.append(frames[i][first_quater_index:last_quater_index])
#             framed_signal.append(np.average(e[i][last_quater_index: end], e[i + 1][0: first_quater_index]))
#             framed_signal.append(e[i][last_quater_index:end])
#         elif i == len(e):
#             framed_signal.append(e[i][first_quater_index:end])
#
# print(framed_signal)


# print(mean(1, 3))