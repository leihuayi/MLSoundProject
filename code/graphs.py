#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_CSV = os.path.join(os.path.dirname(__file__),"../data/train.csv")
TRAIN_PART_CSV = os.path.join(os.path.dirname(__file__),"../data/train-sample.csv")
TRAIN_AUDIO_PATH = "../data/audio_train/"

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   load_sound_files()                                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Returns array of raw sounds loaded from librosa.                                            #
#                                                                                               #
#***********************************************************************************************#
def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   plot_waves()                                                                                #
#                                                                                               #
#   Description:                                                                                #
#   Pots sound waves for 4 sound_names                                                          #
#                                                                                               #
#***********************************************************************************************#
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

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   plot_specgram()                                                                             #
#                                                                                               #
#   Description:                                                                                #
#   Pots sound spectrogram for 4 sounds                                                         #
#                                                                                               #
#***********************************************************************************************#    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    fig.subplots_adjust(top=0.9)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,2,i)
        plt.specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram",fontsize=15)
    plt.show()

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   plot_specgram()                                                                             #
#                                                                                               #
#   Description:                                                                                #
#   Pots sound log spectrogram for 4 sounds                                                     #
#                                                                                               #
#***********************************************************************************************#    
def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    fig.subplots_adjust(top=0.9)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(2,2,i)
        mfcc = librosa.feature.mfcc(wav, sr = SAMPLE_RATE, n_mfcc=40)
        mfcc.shape
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram",fontsize=15)
    plt.show()

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   plot_mfcc()                                                                                 #
#                                                                                               #
#   Description:                                                                                #
#   Pots MFCC                                                                                   #
#                                                                                               #
#***********************************************************************************************#    
def plot_mfcc(sound_names,sound_file_paths):
    i = 1
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    fig.subplots_adjust(top=0.9)
    for n,f in zip(sound_names,sound_file_paths):
        X, sample_rate = librosa.load(f)
        mfcc = librosa.feature.mfcc(X, sr = sample_rate, n_mfcc=40)
        mfcc.shape
        plt.subplot(2,2,i)
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 4: MFCC",fontsize=15)
    plt.show()


#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   plot_categories()                                                                           #
#                                                                                               #
#   Description:                                                                                #
#   Pots samples repartition in plot_categories                                                 #
#                                                                                               #
#***********************************************************************************************#    
def plot_categories():
    train = pd.read_csv(TRAIN_CSV)
    category_group = train.groupby(['label', 'manually_verified']).count()
    fig, axes = plt.subplots()
    axes = category_group.unstack().reindex(category_group.unstack().sum(axis=1).sort_values().index)\
              .plot(kind='bar', stacked=True, title="Number of Audio Samples per Category", figsize=(16,10))
    axes.set_xlabel("Category")
    axes.set_ylabel("Number of Samples")
    plt.show()


#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   main function                                                                               #
#                                                                                               #
#***********************************************************************************************#  
def main():
    data = pd.read_csv(TRAIN_PART_CSV)

    sound_file_paths = [TRAIN_AUDIO_PATH+f for f in data["fname"]]
    sound_names = data["label"].tolist()

    raw_sounds = load_sound_files(sound_file_paths)
    
    #plot_waves(sound_names,raw_sounds)
    #plot_specgram(sound_names,raw_sounds)
    #plot_log_power_specgram(sound_names,raw_sounds)
    #plot_mfcc(sound_names, sound_file_paths)
    #plot_categories()

    

main()