#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

TRAIN_CSV = os.path.join(os.path.dirname(__file__),"../data/train.csv")
TRAIN_AUDIO_PATH = os.path.join(os.path.dirname(__file__),"../data/audio_train/")

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    fig.subplots_adjust(top=0.9)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,2,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",fontsize=15)
    plt.show()
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    fig.subplots_adjust(top=0.9)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,2,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram",fontsize=15)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    fig.subplots_adjust(top=0.9)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,2,i)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(f))**2, ref=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram",fontsize=15)
    plt.show()

def main():
    sound_file_paths = ["../data/audio_train/"+f for f in os.listdir(TRAIN_AUDIO_PATH)]

    sound_names = []

    with open(TRAIN_CSV, 'r') as f:
        lines = f.readlines()
        for l in lines[1:]:
            sound_names.append(l.split(",")[1])

    print(sound_file_paths)
    print(sound_names)

    raw_sounds = load_sound_files(sound_file_paths)

    plot_waves(sound_names,raw_sounds)
    plot_specgram(sound_names,raw_sounds)
    plot_log_power_specgram(sound_names,raw_sounds)

main()
