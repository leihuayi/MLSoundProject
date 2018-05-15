#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import utils
import features
import train

# Define global parameters to be used through out the program
TRAIN_CSV = "../data/train.csv"
TRAIN_AUDIO_PATH = "../data/audio_train/"
TEST_AUDIO_PATH = "../data/audio_test"

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
    # create a dictionary from the provided train.csv file
    dictionary = utils.create_dictionary(TRAIN_CSV)
    
    # call the feature extraction module to get audio features
    tr_features, tr_labels, tr_verified =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary)
    ts_features, ts_name_list = features.parse_audio_files_predict(TEST_AUDIO_PATH)
    
    # dunno what is going on here.
    tr_labels = features.one_hot_encode(tr_labels)
    
    # use the above extracted features for the training of the model
    train.multilayer_neural_network(tr_features, tr_labels, ts_features, ts_name_list, dictionary)
    

# call the main program.
main()
