##############################          LINKS           #################################


# https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

##########################################################################################

import main
from main import audio, sample_rate
# from python_speech_features import mfcc
import python_speech_features
import numpy

print(audio.shape)

# frame 20ms calculate
frame_time = 0.02

# obliczenie nakładkowania ramek czasowych (25%)
winstep = 0.25 * frame_time

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
lifter = 22

# wspolczynniki cepstralne - jesli true to zamienia na logarytmiczne o takiej samej energii
log_or_not = True

# okno
window = numpy.hamming




# mfcc calculation
def getMFCC():
    MFCC = python_speech_features.base.mfcc(audio, samplerate=sample_rate, winlen=frame_time, winstep=winstep,
                                            numcep=numcep,
                                            nfilt=filters_amount, nfft=512, lowfreq=lowfreq, highfreq=highfreq,
                                            preemph=preemph, ceplifter=lifter,
                                            appendEnergy=log_or_not, winfunc=window)
    return MFCC

def function():
    pass


  #297x13 to oznacza wiec mamy 297 ramek (160 ktore nachodza na siebie (25% długości nachodzenia)


def getDeltas():
    return python_speech_features.base.delta(MFCC, MFCC.shape[0])


MFCC = getMFCC()
Deltas = getDeltas()

print(Deltas.shape)
print('asd')


