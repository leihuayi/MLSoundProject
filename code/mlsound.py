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
#TEST_CSV = "../data/train.csv"
#TEST_AUDIO_PATH = "../data/audio_train"

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
    f, labels, verified =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary)
    
    # use the above extracted features for the training of the model
    train.multilayer_neural_network(f, labels, dictionary)
    
    #for x in dictionary:
    #   print(x, dictionary[x])

# call the main program.
main()
