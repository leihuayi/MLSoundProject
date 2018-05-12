#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import os
import librosa
import numpy as np

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   extract_features()                                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Extracts features from an audio file using the librosa library.                             #
#                                                                                               #
#***********************************************************************************************#
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   parse_audio_files_predict()                                                                 #
#                                                                                               #
#   Description:                                                                                #
#   Parses audio data that needs to be predicted upon.                                          #
#                                                                                               #
#***********************************************************************************************#
def parse_audio_files_predict(audio_path, test_csv_path, file_ext="*.wav"):
    # initialize variables
    features = np.empty((0,193))
    # read audio files and extract features
    with open(os.path.join(os.path.dirname(__file__), test_csv_path), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            splits = line.split(",")
            fn = audio_path+splits[0]
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
    
    # return the extracted features to the calling program
    return np.array(features)

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   parse_audio_files_train()                                                                   #
#                                                                                               #
#   Description:                                                                                #
#   Parses the audio data that is to be used for training.                                      #
#                                                                                               #
#***********************************************************************************************#
def parse_audio_files_train(audio_path, train_csv_path, label_dictionary, file_ext="*.wav"):
    # initialize variables
    features, labels, verified = np.empty((0,193)), np.empty(0), np.empty(0)    
    # read audio files and extract features    
    with open(os.path.join(os.path.dirname(__file__), train_csv_path), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            splits = line.split(",")
            fn = audio_path+splits[0]
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label_dictionary[splits[1]])
            if splits[2].rstrip() == "1":
                verified = np.append(verified, True)    
            else:
                verified = np.append(verified, False)
    # return the extracted features to the calling program
    return np.array(features), np.array(labels, dtype = np.int), np.array(verified, dtype=np.bool)
