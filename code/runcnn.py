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
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),"../data/submission.csv")

# Run convolutional neural network only
def run_cnn():
    # get features
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
