#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import os
import librosa
import utils
import json
import pickle
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
TR_FEATURE_NPY = os.path.join(os.path.dirname(__file__),"../data/f1.npy")
TR_LABEL_NPY = os.path.join(os.path.dirname(__file__),"../data/f2.npy")
TR_VERIFY_NPY = os.path.join(os.path.dirname(__file__),"../data/f3.npy")
TS_FEATURE_NPY = os.path.join(os.path.dirname(__file__),"../data/f4.npy")
TS_F_NAME_NPY = os.path.join(os.path.dirname(__file__),"../data/f5.npy")
LABEL_DICT_NPY = os.path.join(os.path.dirname(__file__),"../data/f6.txt")

CHUNK_SIZE = 500
FEATURE_SIZE = 193

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
    features, labels, verified = np.empty((0,FEATURE_SIZE)), np.empty(0), np.empty(0)    
    # process this threads share of workload
    for i in range(data.shape[0]):
            # add a log message to be displayed after processing every 250 files.
            if i%250 == 0:
                utils.write_log_msg("FEATURE_TRAIN - {0}...".format(i))
            line = data.iloc[i]
            fn = audio_path+line["fname"]

            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
            #ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,mfccs])
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
    features = np.empty((0,FEATURE_SIZE))
    # traverse through the name list and process this threads workload
    for fname in name_list:
        X, sample_rate = librosa.load(audio_path+fname, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

        #mfccs, chroma, mel, contrast,tonnetz = extract_feature(audio_path+fname)
        #ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,mfccs])
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
    features = np.empty((0,FEATURE_SIZE))
    
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

    # normalize data
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean)/std
    
    # perform final touches to extracted arrays
    features = np.array(features)

    # normalize data
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean)/std
    
    # return the extracted features to the calling program
    return features, name_list

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
    features, labels, verified = np.empty((0,FEATURE_SIZE)), np.empty(0), np.empty(0)    
    
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

    # normalize data
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean)/std
    
    # perform final touches to extracted arrays
    features = np.array(features)
    labels = one_hot_encode(np.array(labels, dtype = np.int))
    verified = np.array(verified, dtype=np.bool)

    # normalize data
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean)/std
    
    # return the extracted features to the calling program
    return features, labels, verified

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

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   read_features()                                                                             #
#                                                                                               #
#   Description:                                                                                #
#   Program responsible for reading pre-made features files without any extraction.             #
#                                                                                               #
#***********************************************************************************************#
def read_features():
    tr_features = np.load(TR_FEATURE_NPY)
    tr_labels = np.load(TR_LABEL_NPY)
    ts_features = np.load(TS_FEATURE_NPY)
    tr_verified = np.load(TR_VERIFY_NPY)
    dictionary = json.load(open(LABEL_DICT_NPY))
    ts_name_list = pickle.load(open(TS_F_NAME_NPY, "rb"))
    return dictionary, tr_features, tr_labels, tr_verified, ts_features, ts_name_list

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   store_features()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Program responsible for storing pre-made features to files.                                 #
#                                                                                               #
#***********************************************************************************************#
def store_features(dictionary, tr_features, tr_labels, tr_verified, ts_features, ts_name_list):
    np.save(TR_FEATURE_NPY, tr_features)
    np.save(TR_LABEL_NPY, tr_labels)
    np.save(TS_FEATURE_NPY, ts_features)
    np.save(TR_VERIFY_NPY, tr_verified)
    json.dump(dictionary, open(LABEL_DICT_NPY,'w'))
    pickle.dump(ts_name_list, open(TS_F_NAME_NPY, "wb"))
