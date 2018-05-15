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


def create_label_dictionary(csv_path):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path))
    labelList = df.groupby(["label"]).count().index.get_level_values("label").tolist()
    for i in range(len(labelList)):
        dictionary[labelList[i]] = i
    print(dictionary)

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
<<<<<<< HEAD
    
    # call the feature extraction module to get audio features
    tr_features, tr_labels, tr_verified =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary)

    #dictionary = utils.create_dictionary(TEST_CSV)
    ts_features, ts_labels, ts_verified = features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary)
    
    # dunno what is going on here.
    tr_labels = features.one_hot_encode(tr_labels)
    ts_labels = features.one_hot_encode(ts_labels)
=======
    # call the feature extraction module to get audio features
    tr_features, tr_labels, tr_verified =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,TRAIN_CSV,dictionary)

    dictionary = utils.create_dictionary(TEST_CSV)
	ts_features, ts_labels, ts_verified = features.parse_audio_files_train(TEST_AUDIO_PATH,TEST_CSV,dictionary)

	tr_labels = one_hot_encode(tr_labels)
	ts_labels = one_hot_encode(ts_labels)
>>>>>>> 9e0e57390659370b3518ecdb8e5e6bcc7cb42c26
    
    # use the above extracted features for the training of the model
    train.multilayer_neural_network(tr_features, tr_labels, ts_features, ts_labels, dictionary)
    

# call the main program.
main()
