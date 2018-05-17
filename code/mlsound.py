#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import os
import utils
import features
import train

# Define global parameters to be used through out the program
TRAIN_CSV = os.path.join(os.path.dirname(__file__),"../data/train.csv")
TRAIN_AUDIO_PATH = os.path.join(os.path.dirname(__file__),"../data/audio_train/")
TEST_AUDIO_PATH = os.path.join(os.path.dirname(__file__),"../data/audio_test")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),"../data/submission.csv")


#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   main()                                                                                      #
#                                                                                               #
#   Description:                                                                                #
#   Main program responsible for bringing everything together.                                  #
#                                                                                               #
#***********************************************************************************************#
def main():
    # print a log message for status update
    print("creating data dictionary...")
    
    # create a dictionary from the provided train.csv file
    dictionary = utils.create_dictionary(TRAIN_CSV)
    
    # print a log message for status update
    print("extracting features of training data...")
    
    # call the feature extraction module to get audio features
    tr_features, tr_labels, tr_verified =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary)
    
    # print a log message for status update
    print("extracting features of prediction data...")
    
    # call the feature extraction module to get audio features
    ts_features, ts_name_list = features.parse_audio_files_predict(TEST_AUDIO_PATH)
    
    # print a log message for status update
    print("transforming labels of the training data...")
    
    # dunno what is going on here.
    tr_labels = features.one_hot_encode(tr_labels)

    # print a log message for status update
    print("starting multi-layer neural network training...")
    
    # use the above extracted features for the training of the model
    predict_multilayer_nn = train.multilayer_neural_network(tr_features, tr_labels, ts_features, n_classes=len(dictionary))
    
    # print a log message for status update
    print("outputing prediction results to a csv file...")
    
    # print the predicted results to a csv file.
    utils.print_csv_file(predict_multilayer_nn, ts_name_list, dictionary, OUTPUT_CSV)

    # print a log message for status update
    print("done...")

# call the main program.
main()
