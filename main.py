
import scipy.signal as sig
from scipy.io import wavfile
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import numpy as np
import scipy

# audio read
sample_rate, audio = wavfile.read("./jeden_16bit.wav")


# FFT length
frame_time = 0.02
nftt = frame_time * sample_rate
# print(nftt)


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
            sample_position += int(0.75 * frame_size)

    frames = list(frames.values())
    # print(frames)
    windowed_frames = window_frames(frames)

    # asd = []
    # first_frame = frames[0]
    # asd.append(first_frame[0:int(0.75* len(first_frame))])
    # asd.append(np.mean(np.array(first_frame[(int(0.75*len(first_frame))): len(first_frame)]), np.array(first_frame[(int(len(first_frame))): 0.25 * len(first_frame)])))

    framed_signal = list()

    first_quater_index = int(np.floor(0.25 * len(windowed_frames[0])))
    last_quater_index = int(np.floor(0.75 * len(windowed_frames[0])))
    # print(first_quater_index, last_quater_index)
    end = len(windowed_frames[0])

    for i in range(0, len(windowed_frames)-1):
        #dla pierwszego elementu
        if not i+1 > len(windowed_frames):
            if i == 0:
                framed_signal.append(windowed_frames[i][0:last_quater_index])
                # framed_signal.append(frames[i][last_quater_index : end] + frames[i+1][0: first_quater_index])
            elif i != len(windowed_frames):
                framed_signal.append(np.add(windowed_frames[i][last_quater_index: end], windowed_frames[i + 1][0: first_quater_index]))
                framed_signal.append(windowed_frames[i][first_quater_index:last_quater_index])
            elif i == len(windowed_frames):
                framed_signal.append(windowed_frames[i][first_quater_index:end])


    # framed_signal_ = [item for sublist in windowed_frames for item in sublist]
    framed_signal_ = [item for sublist in framed_signal for item in sublist]

    # print(len(framed_signal_))
    # print(len(audio))
    return framed_signal_


def window_frames(frames):
    plt.figure(1)
    plt.title("example frame vs windowed example frame")
    plt.plot(frames[30])

    for i in range(0, len(frames)):
        array1 = np.array(frames[i])
        array2 = np.hamming(len(frames[i]))

        frames[i] = array1 * array2

    plt.plot(frames[30])
    # plt.show()
    return frames


def getFFT(signal):
    FFT = fft.fft(signal, 320)
    FFT = FFT[:len(FFT) // 2]
    return FFT


def showFFT(signal):
    FFT = getFFT(signal)
    plt.plot(FFT[:len(FFT) // 2])
    plt.show()


def high_pass_filter(signal):
    b, a = sig.butter(1, 0.8, 'high', analog=True)
    [W, h] = sig.freqz(b, a, worN=nftt)
    W = sample_rate * W / 2 * np.pi
    filtered_signal = sig.lfilter(b, a, signal)
    return filtered_signal


#######################################

#######################################


def trigger_plots(filtered_audio, framed_audio):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title("unfiltered vs filtered signal")
    plt.plot(audio)
    plt.subplot(2, 1, 2)
    plt.plot(filtered_audio)
    plt.show()

    # FFT przed vs po filtracji
    # plt.figure(2)
    # plt.subplot(2, 1, 1)
    # plt.title("FFT of unfiltered vs filtered signal")
    # plt.plot(np.abs(getFFT(audio)))
    # plt.subplot(2, 1, 2)
    # plt.plot(np.abs(getFFT(filtered_audio)))
    # plt.show()

    # sygnal po filtracji vs sygnal po filtracji zramkoway i zokienkowany
    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.title("before vs after framing and windowing ")
    plt.plot(filtered_audio)
    plt.subplot(2, 1, 2)
    plt.plot(framed_audio)
    plt.show()

    plt.figure(4)
    show_bank()


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
    for el in freqs:
        rounded_freqs.append(np.floor((nftt + 1) * el/sample_rate))

    return rounded_freqs


# def calculate_filter_middle_freqs(low_freq, high_freq, filters_number):
#     low_mel_freq = freq_to_mel(low_freq)
#     high_mel_freq = freq_to_mel(high_freq)
#     freqs = list()
#     mel_freqs_difference = high_mel_freq - low_mel_freq
#     mel_freqs_step = mel_freqs_difference/(filters_number + 1)
#
#     for el in range(0, filters_number + 1):
#         freqs.append(low_mel_freq + el * mel_freqs_step)
#
#     freqs = mel_to_freq(freqs)
#     freqs = np.floor(freqs)
#     freqs = round_freqs(freqs)
#     print(freqs)
#     return freqs

def calculate_filter_middle_freqs(low_freq, high_freq, filters_number):
    low_mel_freq = freq_to_mel(low_freq)
    high_mel_freq = freq_to_mel(high_freq)
    mel_points = np.linspace(low_mel_freq, high_mel_freq, filters_number + 2)

    freqs = mel_to_freq(mel_points)
    freqs = np.floor(freqs)
    freqs = round_freqs(freqs)
    return freqs


def generate_filter_bank():
    low_freq = 80
    high_freq = 4000
    filters_number = 20

    middle_freqs = calculate_filter_middle_freqs(low_freq, high_freq, filters_number)
    fbank = np.zeros((filters_number, int(np.floor(nftt))))
    for m in range(1, filters_number + 1):
        f_m_minus = int(middle_freqs[m - 1])
        f_m = int(middle_freqs[m])
        f_m_plus = int(middle_freqs[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - middle_freqs[m - 1]) / (middle_freqs[m] - middle_freqs[m - 1])

        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (middle_freqs[m + 1] - k) / (middle_freqs[m + 1] - middle_freqs[m])


    return fbank


def show_bank():
    fbank = generate_filter_bank()
    plt.figure(4)
    plt.title("Mel filter bank")
    plt.plot(fbank.T)
    plt.show()


def get_MFCC(audio):
    audio = normalize_signal(audio)
    audio = np.mean(audio, axis=1)
    filtered_audio = high_pass_filter(audio)
    framed_audio = frame_and_window_signal(filtered_audio)

    fft = np.abs(getFFT(framed_audio))
    fft = np.square(fft)
    filter_bank = generate_filter_bank()

    bands_energies = list()

    for i in range(0, len(filter_bank)):
        for j in range(0, len(filter_bank[0])):
            if filter_bank[i][j] == 0:
                filter_bank[i][j] = 1

    for el in filter_bank:
        bands_energies.append(el * fft)

    bands_energies = 10 * np.log10(bands_energies)

    MFCC = scipy.fftpack.dct(bands_energies)
    MFCC = MFCC[0:13]
    return MFCC


MFCC = get_MFCC(audio)

print("our MFCC: ", MFCC)
print(MFCC.shape)

