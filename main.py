import os
import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
# import IPython.display as ipd
import matplotlib.pyplot as plt

import numpy as np


project_path = 'C:/Users/Adam/PycharmProjects/interfejs_glosowy_projekt/'

# audio read
sample_rate, audio = wavfile.read(project_path + "jeden_16bit.wav")


def normalize_signal(audio):
    audio = audio / np.max(np.abs(audio))
    return audio


def frame_and_window_signal(signal, frame_size=0.02):
    frame_size = int(frame_size * sample_rate)  # number of samples in frame
    frames = {}
    sample_position = 0
    for el in signal:
        if not sample_position > len(signal):
            frames[el] = signal[sample_position: int(sample_position + frame_size)]
            sample_position += int(1 * frame_size)

    frames = list(frames.values())

    windowed_frames = window_frames(frames)

    # asd = []
    # first_frame = frames[0]
    # asd.append(first_frame[0:int(0.75* len(first_frame))])
    # asd.append(np.mean(np.array(first_frame[(int(0.75*len(first_frame))): len(first_frame)]), np.array(first_frame[(int(len(first_frame))): 0.25 * len(first_frame)])))

    framed_signal = [item for sublist in windowed_frames for item in sublist]
    return framed_signal


def window_frames(frames):
    plt.figure(1)
    plt.title("example frame vs windowed example frame")
    plt.plot(frames[30])

    for i in range(0, len(frames)):
        array1 = np.array(frames[i])
        array2 = np.hamming(len(frames[i]))

        frames[i] = array1 * array2

    plt.plot(frames[30])
    plt.show()
    return frames


def getFFT(signal):
    FFT = fft.fft(signal)
    FFT = FFT[:len(FFT) // 2]
    return FFT


def showFFT(signal):
    FFT = getFFT(signal)
    plt.plot(FFT[:len(FFT) // 2])
    plt.show()


def high_pass_filter(signal):
    b, a = sig.butter(6, 0.4, 'high', analog=True)
    [W, h] = sig.freqz(b, a, worN=1024)
    W = sample_rate * W / 2 * np.pi
    filtered_signal = sig.lfilter(b, a, signal)
    return filtered_signal


#######################################
audio = normalize_signal(audio)
audio = np.mean(audio, axis=1)
filtered_audio = high_pass_filter(audio)
framed_audio = frame_and_window_signal(filtered_audio)
FFT_audio = getFFT(framed_audio)
#######################################


def trigger_plots():
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title("unfiltered vs filtered signal")
    plt.plot(audio)
    plt.subplot(2, 1, 2)
    plt.plot(filtered_audio)

    plt.show()

    # FFT przed vs po filtracji
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.title("FFT of unfiltered vs filtered signal")
    plt.plot(getFFT(audio))
    plt.subplot(2, 1, 2)
    plt.plot(getFFT(filtered_audio))
    plt.show()

    # sygnal po filtracji vs sygnal po filtracji zramkoway i zokienkowany
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.title("before vs after framing and windowing ")
    plt.plot(filtered_audio)
    plt.subplot(2, 1, 2)
    plt.plot(framed_audio)
    plt.show()


trigger_plots()
