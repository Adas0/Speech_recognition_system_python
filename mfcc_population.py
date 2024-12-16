from scipy.io import wavfile
import glob
import os

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

