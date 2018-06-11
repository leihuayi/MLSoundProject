#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import utils
import features
import train
import os

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
TRAIN_CSV = os.path.join(os.path.dirname(__file__),"../data/train.csv")
TRAIN_AUDIO_PATH = os.path.join(os.path.dirname(__file__),"../data/audio_train/")
TEST_AUDIO_PATH = os.path.join(os.path.dirname(__file__),"../data/audio_test/")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),"../data/submission.csv")

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   read_audio_files()                                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Program responsible for reading audio files and performing feature extraction on them.      #
#                                                                                               #
#***********************************************************************************************#
def read_audio_files():
    
    # print a log message for status update
    utils.write_log_msg("creating data dictionary...")  
    
    # create a dictionary from the provided train.csv file
    dictionary = utils.create_dictionary(TRAIN_CSV)  
    
    # print a log message for status update
    utils.write_log_msg("extracting features of training data...")  
    # call the feature extraction module to get audio features
    tr_mnn_features, tr_mnn_labels =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary, 0)  
    # call the feature extraction module to get audio features
    tr_cnn_features, tr_cnn_labels =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary, 1)  
    
    # print a log message for status update
    utils.write_log_msg("processed {0} files of training data for mnn...".format(len(tr_mnn_features)))  
    # print a log message for status update
    utils.write_log_msg("processed {0} files of training data for cnn...".format(len(tr_cnn_features)))  
    
    # print a log message for status update
    utils.write_log_msg("extracting features of prediction data...")  
    # call the feature extraction module to get audio features
    ts_mnn_features, ts_mnn_name_list = features.parse_audio_files_predict(TEST_AUDIO_PATH,os.listdir(TEST_AUDIO_PATH), 0)  
    # call the feature extraction module to get audio features
    ts_cnn_features, ts_cnn_name_list = features.parse_audio_files_predict(TEST_AUDIO_PATH,os.listdir(TEST_AUDIO_PATH), 1)  
    
    # print a log message for status update
    utils.write_log_msg("processed {0} files of prediction data for mnn...".format(len(ts_mnn_features)))
    # print a log message for status update
    utils.write_log_msg("processed {0} files of prediction data for cnn...".format(len(ts_cnn_features)))
    
    # print a log message for status update
    utils.write_log_msg("storing features for future use...")
    # store features so that they can be used in future
    features.store_features(dictionary, tr_mnn_features, tr_mnn_labels, ts_mnn_features, ts_mnn_name_list,
                                        tr_cnn_features, tr_cnn_labels, ts_cnn_features, ts_cnn_name_list)
    
    # return the results to calling program
    return dictionary, tr_mnn_features, tr_mnn_labels, ts_mnn_features, ts_mnn_name_list, tr_cnn_features, tr_cnn_labels, ts_cnn_features, ts_cnn_name_list

# Run convolutional neural network only
def run_cnn(_load = False):
    # intialize the log file for current run of the code
    utils.initialize_log()  
    # read audio files and parse them or simply load from pre-extracted feature files
    if _load:
        dictionary, tr_mnn_features, tr_mnn_labels, ts_mnn_features, ts_mnn_name_list, tr_cnn_features, tr_cnn_labels, ts_cnn_features, ts_cnn_name_list  = read_audio_files()  
    else:
        dictionary, tr_mnn_features, tr_mnn_labels, ts_mnn_features, ts_mnn_name_list, tr_cnn_features, tr_cnn_labels, ts_cnn_features, ts_cnn_name_list  = features.read_features()
    # call the 2d convolutional network code here
    cnn_2d_probs = train.keras_convolution_2D(tr_cnn_features, tr_cnn_labels, ts_cnn_features, len(dictionary), training_epochs=50)
    # get top three predictions
    top3 = cnn_2d_probs.argsort()[:,-3:][:,::-1]
    # print the predicted results to a csv file.
    utils.print_csv_file(top3, ts_mnn_name_list, dictionary, OUTPUT_CSV)
    

def main():
	utils.write_log_msg("Run CNN code ...")
	#limitedCsv()
	run_cnn()

main()
