from scipy.io import wavfile
import glob
import os
# from main import

class WavFile:
    def __init__(self):
        self.sample_rate = []
        self.audio = []
        self.path = []

    def read_audio(self, path):
        self.sample_rate, self.audio = wavfile.read(path)


class WordPattern:
    def __init__(self, files):
        self.file = files

    def get_files(self, word):
        pass

    def get_pattern_mfcc(self):
        files = self.get_files()
        pass


# files_zima = list()
# os.chdir("./pory_roku")
# for file in glob.glob("zima*.wav"):
#     files_zima.append(file)
#
# num_ppl_in_base = 6
#
# # files[0].path = "./pory_roku/Wiosna-Magda-Ceglarek.wav"
#
# print("asd" + files_zima[0])
