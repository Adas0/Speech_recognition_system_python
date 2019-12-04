import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
# import IPython.display as ipd
import matplotlib.pyplot as plt


project_path = 'C:/Users/Adam/PycharmProjects/interfejs_glosowy_projekt/'

# read wav file - we get it's sample_rate and the file(samples)
sample_rate, audio = wavfile.read(project_path + "jeden_16bit.wav")

# get print sample rate and length of the file
print("Sample rate: {0}Hz".format(sample_rate))
print("Audio duration: {0}s".format(len(audio) / sample_rate))


def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio


def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):

    # tworzenie tablicy: audio i FFT_size / 2
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))

    for n in range(frame_num):
        frames[n] = audio[n * frame_len:n * frame_len + FFT_size]

    return frames


normalize_audio(audio)
hop_size = 15 #ms
FFT_size = 2048

print((np.pad(audio, int(FFT_size / 2), mode='reflect')).shape)
frame_len = np.round(sample_rate * hop_size / 1000).astype(int) #jedna liczba
frame_num = int((len(audio) - FFT_size) / frame_len) + 1
frames = np.zeros((frame_num, FFT_size))
# audio = audio[:, 1]
print(audio.shape)
# for n in range(frame_num):
#     frames[n] = audio[n * frame_len:n * frame_len + FFT_size]

    # print(frames[n])

# audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
# print("Framed audio shape: {0}".format(audio_framed.shape))