#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import os
import utils
import features
import train

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
    tr_features, tr_labels, tr_verified =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary,1)  
    # print a log message for status update
    utils.write_log_msg("processed {0} files of training data...".format(len(tr_features)))  
    # print a log message for status update
    utils.write_log_msg("extracting features of prediction data...")  
    # call the feature extraction module to get audio features
    ts_features, ts_name_list = features.parse_audio_files_predict(TEST_AUDIO_PATH,1)  
    # print a log message for status update
    utils.write_log_msg("processed {0} files of prediction data...".format(len(ts_features)))
    # print a log message for status update
    utils.write_log_msg("storing features for future use...")
    # store features so that they can be used in future
    features.store_features(dictionary, tr_features, tr_labels, tr_verified, ts_features, ts_name_list)
    # return the results to calling program
    return dictionary, tr_features, tr_labels, tr_verified, ts_features, ts_name_list

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   main()                                                                                      #
#                                                                                               #
#   Description:                                                                                #
#   Main program responsible for bringing everything together.                                  #
#                                                                                               #
#***********************************************************************************************#
def main(_load = True):
    # intialize the log file for current run of the code
    utils.initialize_log()  
    # read audio files and parse them or simply load from pre-extracted feature files
    if _load:
        dictionary, tr_features, tr_labels, tr_verified, ts_features, ts_name_list = read_audio_files()  
    else:
        dictionary, tr_features, tr_labels, tr_verified, ts_features, ts_name_list = features.read_features()
    # print a log message for status update
    utils.write_log_msg("starting multi-layer neural network training...")
    # use the above extracted features for the training of the model
    predictions_top3 = train.train(tr_features, tr_labels, ts_features, n_classes=len(dictionary))
    # print a log message for status update
    utils.write_log_msg("outputing prediction results to a csv file...")
    # print the predicted results to a csv file.
    utils.print_csv_file(predictions_top3, ts_name_list, dictionary, OUTPUT_CSV)
    # print a log message for status update
    utils.write_log_msg("done...")

# call the main program.
main()
