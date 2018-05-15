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
TEST_CSV = "../data/verifying.csv"
TEST_AUDIO_PATH = "../data/audio_verifying"

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

    #dictionary = utils.create_dictionary(TEST_CSV)
    ts_features, ts_labels, ts_verified = features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary)
    
    # dunno what is going on here.
    tr_labels = features.one_hot_encode(tr_labels)
    ts_labels = features.one_hot_encode(ts_labels)
    
    # use the above extracted features for the training of the model
    train.multilayer_neural_network(tr_features, tr_labels, ts_features, ts_labels, dictionary)
    

# call the main program.
main()
