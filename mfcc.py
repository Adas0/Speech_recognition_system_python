##############################          LINKS           #################################


# https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# https://python-speech-features.readthedocs.io/en/latest/

#Date: 2018
#Author: Adam Korytowski

##########################################################################################

import main
from main import audio, sample_rate
import python_speech_features
import numpy
import numpy as np
from statistics import mean

# frame 20ms calculate
frame_time = 0.02
print(len(audio))
print(len(audio)/(frame_time * sample_rate))

# obliczenie nakładkowania ramek czasowych (25%)
winstep = 0.02

# liczba wspolczynnikow mfcc
numcep = 13

# liczba filtrow (14-20)
filters_amount = 20

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
                                            numcep=13, nfilt=20, nfft=64, lowfreq=0, highfreq=None, preemph=0.97,
                                            ceplifter=22, appendEnergy=True, winfunc=numpy.hamming)
    return MFCC

def function():
    pass

MFCC = getMFCCa()
print("mfcc shape", MFCC.shape)
print(MFCC[0])
