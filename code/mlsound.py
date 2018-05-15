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
TEST_CSV = "../data/test.csv"
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
    dictionary = utils.create_dictionary(TRAIN_CSV,1)
    
    # call the feature extraction module to get audio features
    tr_features, tr_labels, tr_verified =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary)

    dictionary = utils.create_dictionary(TEST_CSV,1)
	ts_features, ts_labels_ts_verified = features.parse_audio_files_train(TEST_AUDIO_PATH,TEST_CSV,dictionary)

	tr_labels = one_hot_encode(tr_labels)
	ts_labels = one_hot_encode(ts_labels)
    
    # use the above extracted features for the training of the model
    train.multilayer_neural_network(tr_features, tr_labels, ts_features, ts_labels, dictionary)
    
    #for x in dictionary:
    #   print(x, dictionary[x])

# call the main program.
main()
