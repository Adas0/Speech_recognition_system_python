from numpy.linalg import norm
import scipy.signal as sig
from scipy.io import wavfile
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import numpy as np
import scipy
import glob
import os
from dtw import dtw


sample_rate, audio_wiosna = wavfile.read("./wiosna-Adam-Korytowski.wav")
sample_rate_, audio_lato = wavfile.read("./zima-Adam-Korytowski.wav")
sample_rate__, audio_jesien = wavfile.read("./jesien-Adam-Korytowski.wav")
# sample_rate, audio = wavfile.read("./zima-Adam-Korytowski.wav")


frame_time = 0.02
nftt = frame_time * sample_rate


def normalize_signal(audio_):
    audio_ = audio_ / np.max(np.abs(audio_))
    return audio_


def frame_and_window_signal(signal, frame_time=0.02):
    frame_size = int(frame_time * sample_rate)  # number of samples in frame
    frames = {}
    sample_position = 0
    for el in signal:
        if not sample_position > len(signal):
            frames[el] = signal[sample_position: int(sample_position + frame_size)]
            sample_position += int(0.75 * frame_size)

    frames = list(frames.values())

    windowed_frames = window_frames(frames)
    # fft_frames = fft.fft(windowed_frames)
    fft_frames = list()
    for el in windowed_frames:
        fft_frames.append(getFFT(el))

    framed_signal = list()

    first_quater_index = int(np.floor(0.25 * len(windowed_frames[0])))
    last_quater_index = int(np.floor(0.75 * len(windowed_frames[0])))
    end = len(windowed_frames[0])

    for i in range(0, len(windowed_frames)-1):
        if not i+1 > len(windowed_frames):
            if i == 0:
                framed_signal.append(windowed_frames[i][0:last_quater_index])
            elif i != len(windowed_frames):
                last_quater = windowed_frames[i][last_quater_index: end]
                first_quater = windowed_frames[i + 1][0: first_quater_index]
                if not len(last_quater) == len(first_quater):
                    first_quater = list(first_quater)
                    first_quater.append(0)
                    first_quater = np.array(first_quater)
                framed_signal.append(np.add(last_quater, first_quater))
                framed_signal.append(windowed_frames[i][first_quater_index:last_quater_index])
            elif i == len(windowed_frames):
                framed_signal.append(windowed_frames[i][first_quater_index:end])

    framed_signal_ = [item for sublist in framed_signal for item in sublist]
    return framed_signal_, fft_frames


def hamming_coefficients(x):
    hamming_coeffs = np.zeros(x)
    for i in range(0, x):
       hamming_coeffs[i] = 0.54 - 0.46*np.cos(2*np.pi*(i/(x-1)))
    return hamming_coeffs


def window_frames(frames):
    plt.figure(1)
    plt.title("example frame vs windowed example frame")
    plt.plot(frames[30])

    for i in range(0, len(frames)):
        array1 = np.array(frames[i])
        array2 = hamming_coefficients(len(frames[i]))
        frames[i] = array1 * array2

    plt.plot(frames[30])
    # plt.show()
    return frames


def getFFT(signal):
    FFT = fft.fft(signal, 160)
    FFT = FFT[:len(FFT)]
    FFT = np.abs(FFT)
    return FFT


def showFFT(signal):
    FFT = getFFT(signal)
    plt.plot(FFT[:len(FFT) // 2])
    plt.show()


def high_pass_filter(signal):
    b, a = sig.butter(1, 0.8, 'high', analog=True)
    filtered_signal = sig.lfilter(b, a, signal)
    return filtered_signal


#######################################

#######################################


def trigger_plots(filtered_audio, framed_audio):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title("unfiltered vs filtered signal")
    plt.plot(audio_wiosna)
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
        rounded_freqs.append(np.floor((nftt + 1) * 2 * el / sample_rate))

    return rounded_freqs


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
    # fbank = np.zeros(filters_number, int(np.floor(nftt/2 + 1)))
    fbank = np.zeros((filters_number, int(np.floor(nftt ))))
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


def fill_incomplete_frame(frame, complete_frame_len):
    frame = list(frame)
    for i in range(0, complete_frame_len - len(frame)):
        frame.append(0)
    return np.array(frame)


def get_MFCC(audio):
    audio = normalize_signal(audio)
    audio = np.mean(audio, axis=1)
    filtered_audio = high_pass_filter(audio)
    framed_audio, fft_frames = frame_and_window_signal(filtered_audio)

    fft = np.abs(getFFT(framed_audio))
    fft = np.square(fft)

    filter_bank = generate_filter_bank()

    for i in range(0, len(filter_bank)):    # numer filtru
        # next((i for i, x in enumerate(filter_bank[i]) if x), None)
        for j in range(0, len(filter_bank[0])):     # numer elementu w filtrze
            if filter_bank[i][j] == 0:
                filter_bank[i][j] = 1

    mel_filtered_fft = list()

    fft_frames[-1] = fill_incomplete_frame(fft_frames[-1], len(fft_frames[0]))
    new_frames = list()
    for frame in fft_frames:
        filtered = 0
        for el in filter_bank:
            filtered += el * frame
        new_frames.append(filtered)

    # MFCC = new_frames
    # matrix



    # matrix = scipy.fftpack.dct(new_frames)
    # to jest macierz (ramki x probki w tych ramkach). czyli jakieś 263x160. (260 ramek, każda po 160 próbek)
    # ma powstać: 263 x 20. Czyli z tego 160 trzeba zrobić 20. Jak? - Trzeba uśrednić wartości w pasmach melowych:
    # czyli: patrzysz gdzie się zaczyna i kończy pierwszy filtr : zaczyna się w indeksie 4, kończy w 8. więc bierzesz
    # wartości od 4 do 8 i robisz z niech średnią. Ta średnia to pierwszy element. Średnia z drugiego pasma to będzie
    # drugi element, etc, aż do dwudziestego filtra. (Jak otworzysz sobie zmienna filter_bank to tam widac gdzie
    # dokladnie sie zaczynaja i koncza te filtry)
    # ta macierz to bedzie już poprawne mfcc

    new_frames = np.matrix(new_frames)
    MFCC = matrix_sum(new_frames)
    MFCC = 20 * np.log10(MFCC)

    MFCC = scipy.fftpack.dct(MFCC)

    MFCC = np.transpose(MFCC)
    MFCC = MFCC[0:13]
    # trigger_plots(filtered_audio, framed_audio)
    return MFCC


def matrix_sum(matrix):
    new_matrix = np.zeros((len(matrix), 20))
    indexes = [
        [4, 9], [6, 12], [10, 16], [13, 19], [17, 24], [20, 28], [25, 34], [29, 39], [35, 45], [40, 52],
        [46, 59], [53, 67], [60, 75], [68, 85], [76, 95], [86, 106], [96, 118], [107, 131], [119, 145], [132, 159]
    ]

    for k in range(0, len(matrix)):
        for j in range(0, len(indexes)):
            temp_indexes = indexes[j]
            temp_sum = 0
            for i in range(temp_indexes[0], temp_indexes[1]):
                temp_sum += matrix[k, i]
            new_matrix[k, j] = temp_sum
    return new_matrix


def get_mfcc_pattern(word):
    MFCCs = list()
    word_files = list()

    for file in glob.glob(word + "*.wav"):
        _audio = list()
        _audio.clear()
        mfcc = ()
        try:
            word_files.append(file)
        except:
            print("cannot find files")
        _sample_rate, _audio = wavfile.read(file)
        try:
            mfcc = get_MFCC(_audio)
        except:
            print("cannot calculate mfcc")
        MFCCs.append(mfcc)
    sum = 0
    avgerage_mfcc = 0
    for i in range(len(MFCCs)):
        MFCCs[i] = np.matrix(MFCCs[i])
        sum += (np.array(MFCCs[i]))
        avgerage_mfcc = sum/len(MFCCs)
        avgerage_mfcc = np.matrix(avgerage_mfcc)
    return avgerage_mfcc


files_zima = list()
os.chdir("./")

def get_distances(word):

    MFCC_tested_word = get_MFCC(audio_lato)
    word_files = list()
    # dcts = list()
    for file in glob.glob(word + "*.wav"):
        word_files.append(file)

    print(word_files)
    dist = list()
    for el in word_files:
        _sample_rate, _audio = wavfile.read(el)
        dist_, cost, acc_cost, path = dtw(MFCC_tested_word.T, get_MFCC(_audio).T, dist=lambda x, y: norm(x - y, ord=1))
        dist.append(dist_)

    return dist



words = ["wiosna", "lato", "jesień", "zima", "poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]

wiosna = list()
os.chdir("./pory_roku_8k/")
wiosna.append(get_distances("wiosna"))
wiosna.append(get_distances("lato"))
wiosna.append(get_distances("jesień"))
wiosna.append(get_distances("zima"))

os.chdir("..")
os.chdir("./dni_tygodnia_8k/")
wiosna.append(get_distances("poniedziałek"))
wiosna.append(get_distances("wtorek"))
wiosna.append(get_distances("środa"))
wiosna.append(get_distances("czwartek"))
wiosna.append(get_distances("piątek"))
wiosna.append(get_distances("sobota"))
wiosna.append(get_distances("niedziela"))


print(wiosna)

