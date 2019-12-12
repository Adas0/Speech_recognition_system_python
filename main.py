
import scipy.signal as sig
from scipy.io import wavfile
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import numpy as np


project_path = 'C:/Users/Adam/PycharmProjects/interfejs_glosowy_projekt/'

# audio read
sample_rate, audio = wavfile.read(project_path + "jeden_16bit.wav")


def normalize_signal(audio_):
    audio_ = audio_ / np.max(np.abs(audio))
    return audio_


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
    FFT = fft.fft(signal, 512)
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


# ta funkcja poprawnie przelicza, potwierdzone
def freq_to_mel(freq):
    return 1125 * np.log(1 + freq/700)


def mel_to_freq(mel_freqs):
    freqs = list()
    for i in range(len(mel_freqs)):
        freqs.append(700 * (np.exp(mel_freqs[i]/1125)-1))
    return freqs


def round_freqs(freqs):
    rounded_freqs = list()
    nfft = 0.02 + sample_rate
    for el in freqs:
        rounded_freqs.append(np.floor((512 + 1) * el/sample_rate))

    return rounded_freqs


def calculate_filter_middle_freqs():
    low_freq = 80
    high_freq = 4000
    low_mel_freq = freq_to_mel(low_freq)
    high_mel_freq = freq_to_mel(high_freq)
    freqs = list()
    mel_freqs_difference = high_mel_freq - low_mel_freq
    mel_freqs_step = mel_freqs_difference/14

    for el in range(1, 15):
        freqs.append(low_mel_freq + el*mel_freqs_step)

    freqs = mel_to_freq(freqs)
    freqs = round_freqs(freqs)

    return freqs


# def get_mel_filters():
#     middle_freqs = calculate_filter_middle_freqs()
#     filters = list()
#     numer_of_filters = 14
#     for i in range(0, )

# print(freq_to_mel(300), freq_to_mel(8000))

