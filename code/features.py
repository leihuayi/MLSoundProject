#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import os
import glob
import librosa
import utils
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
CHUNK_SIZE = 500

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
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
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
#   p_train_thread()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Internal function to parse training audio files in multi-threaded environment.              #
#                                                                                               #
#***********************************************************************************************#
def p_train_thread(audio_path, label_dictionary, data):
    # initialize variables
    features, labels, verified = np.empty((0,193)), np.empty(0), np.empty(0)    
    # process this threads share of workload
    for i in range(data.shape[0]):
            # add a log message to be displayed after processing every 250 files.
            if i%250 == 0:
                utils.write_log_msg("FEATURE_TRAIN - {0}...".format(i))
            line = data.iloc[i]
            fn = audio_path+line["fname"]
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label_dictionary[line["label"]])
            if line["manually_verified"] == 1:
                verified = np.append(verified, True)    
            else:
                verified = np.append(verified, False)
    # return the extracted features to the calling program
    return features, labels, verified
#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   p_train_thread()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Internal function to parse training audio files in multi-threaded environment.              #
#                                                                                               #
#***********************************************************************************************#
def p_predict_thread(audio_path, name_list):
    # initialize variables
    features = np.empty((0,193))
    # traverse through the name list and process this threads workload
    for fname in name_list:
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(audio_path+fname)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        # add a log message to be displayed after processing every 250 files.
        if len(features)%250 == 0:
            utils.write_log_msg("FEATURE_PREDICT - {0}...".format(len(features)))
    return features

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   parse_audio_files_predict()                                                                 #
#                                                                                               #
#   Description:                                                                                #
#   Parses audio data that needs to be predicted upon.                                          #
#                                                                                               #
#***********************************************************************************************#
def parse_audio_files_predict(audio_path, file_ext="*.wav"):
    
    # initialize variables
    features = np.empty((0,193))
    
    # get the list of files in the audio folder
    name_list = os.listdir(audio_path)
    
    # create a thread pool to process the workload
    thread_pool = []
    
    # split the filename list into chunks of 'CHUNK_SIZE' files each
    data = utils.generate_chunks(name_list, CHUNK_SIZE)
    
    # each chunk is the amount of data that will be processed by a single thread
    for chunk in data:
        thread_pool.append(utils.ThreadWithReturnValue(target=p_predict_thread, args=(audio_path, chunk)))
    
    # print a log message for status update
    utils.write_log_msg("PREDICT: creating a total of {0} threads...".format(len(thread_pool)))  
    
    # start the entire thread pool
    for single_thread in thread_pool:
        single_thread.start()
    
    # wait for thread pool to return their results of processing
    for single_thread in thread_pool:
        ft = single_thread.join()
        features = np.vstack([features,ft])
    
    # return the extracted features to the calling program
    return np.array(features), name_list

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
    
    # read audio files using pandas and split it into chunks of 'CHUNK_SIZE' files each
    data = pd.read_csv(train_csv_path, chunksize=CHUNK_SIZE)
    
    # create a thread pool to process the workload
    thread_pool = []
        
    # each chunk is the amount of data that will be processed by a single thread
    for chunk in data:
        thread_pool.append(utils.ThreadWithReturnValue(target=p_train_thread, args=(audio_path, label_dictionary, chunk)))
    
    # print a log message for status update
    utils.write_log_msg("TRAIN: creating a total of {0} threads...".format(len(thread_pool)))  
    
    # start the entire thread pool
    for single_thread in thread_pool:
        single_thread.start()
    
    # wait for thread pool to return their results of processing
    for single_thread in thread_pool:
        ft, lbl, stat = single_thread.join()
        features = np.vstack([features,ft])
        labels = np.append(labels, lbl)
        verified = np.append(verified, stat)
    
    # return the extracted features to the calling program
    return np.array(features), np.array(labels, dtype = np.int), np.array(verified, dtype=np.bool)

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   one_hot_encode()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Creates a matrix size num_samples x num_labels with (i,j) = 1(sample i has label j)         #
#                                                                                               #
#***********************************************************************************************#
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
