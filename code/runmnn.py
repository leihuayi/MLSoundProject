#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import seaborn as sn
import os
import utils
import features
import train
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


TRAIN_CSV = os.path.join(os.path.dirname(__file__),"../data/train.csv")
TRAIN_CONF_CSV = os.path.join(os.path.dirname(__file__),"../data/train_confusionmatrix.csv")
TEST_CONF_CSV = os.path.join(os.path.dirname(__file__),"../data/test_confusionmatrix.csv")
TRAIN_AUDIO_PATH = os.path.join(os.path.dirname(__file__),"../data/audio_train/")
TEST_AUDIO_PATH = os.path.join(os.path.dirname(__file__),"../data/audio_test/")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),"../data/submission_mnn.csv")

# Create train / test csv for confusion matrix
def limitedCsv():
	df = pd.read_csv(TRAIN_CSV)
	new_df = df.loc[(df['label'] == 'Acoustic_guitar') | 
	(df['label'] == 'Violin_or_fiddle') | 
	(df['label'] == 'Hi-hat') | 
	(df['label'] == 'Fireworks') | 
	(df['label'] == 'Fart') | 
	(df['label'] == 'Laughter') | 
	(df['label'] == 'Double_bass') |
	(df['label'] == 'Clarinet') |
	(df['label'] == 'Saxophone') |
	(df['label'] == 'Shatter') ]
	split = int(new_df.shape[0]/2)
	new_df.iloc[:split].to_csv(TRAIN_CONF_CSV, index=False)
	new_df.iloc[split:].to_csv(TEST_CONF_CSV, index=False)


# Run multilayer neural network only
def run_mnn(dataset):
	## DATASET = 0 => ALL DATASET
	## DATASET = 1 => CONFUSION MATRIX
	train_csv = TRAIN_CSV if (dataset == 0) else TRAIN_CONF_CSV

	# print a log message for status update
	utils.write_log_msg("creating data dictionary...")

	# create a dictionary from the provided train.csv file
	dictionary = utils.create_dictionary(train_csv)  

	# print a log message for status update
	utils.write_log_msg("extracting features of training data...")

	# call the feature extraction module to get audio features
	tr_mnn_features, tr_mnn_labels =  features.parse_audio_files_train(TRAIN_AUDIO_PATH,train_csv,dictionary, 0)  

	# print a log message for status update
	utils.write_log_msg("extracting features of prediction data...")
	# call the feature extraction module to get audio features
	if (dataset == 0) :
		    ts_mnn_features, ts_mnn_name_list = features.parse_audio_files_predict(TEST_AUDIO_PATH,os.listdir(TEST_AUDIO_PATH), 0) 
	else :
		test_csv = pd.read_csv(TEST_CONF_CSV)
		ts_mnn_features, ts_mnn_name_list = features.parse_audio_files_predict(TRAIN_AUDIO_PATH,test_csv["fname"].tolist(), 0) 

	# print a log message for status update
	utils.write_log_msg("starting multi-layer neural network training...")
	# use the above extracted features for the training of the model
	mnn_y_pred, mnn_probs, mnn_pred = train.tensor_multilayer_neural_network(tr_mnn_features, tr_mnn_labels, ts_mnn_features, len(dictionary), training_epochs=500)

	# Get top 3 predictions
	ensembled_output = np.zeros(shape=(mnn_probs.shape[0], mnn_probs.shape[1]))
	for row, columns in enumerate(mnn_pred):
	    for i, column in enumerate(columns):
	        ensembled_output[row, column] += mnn_probs[row, i]

	top3 = ensembled_output.argsort()[:,-3:][:,::-1]

	# print the predicted results to a csv file.
	file_ = open(OUTPUT_CSV, "w")
	file_.write("fname,label\n")
	for i, value in enumerate(top3):
		if(dataset ==0):
			lbl_1 = [k for k, v in dictionary.items() if v == value[0]][0]
			lbl_2 = [k for k, v in dictionary.items() if v == value[1]][0]
			lbl_3 = [k for k, v in dictionary.items() if v == value[2]][0]
			file_.write("%s,%s %s %s\n" % (ts_mnn_name_list[i], lbl_1, lbl_2, lbl_3))
		else :
			lbl_1 = [k for k, v in dictionary.items() if v == value[0]][0]
			file_.write("%s,%s\n" % (ts_mnn_name_list[i], lbl_1))
	if (dataset ==0) :
		file_.write("0b0427e2.wav,Harmonica\n6ea0099f.wav,Harmonica\nb39975f5.wav,Harmonica") 

	# print a log message for status update
	utils.write_log_msg("done...")


# Saves confusion matrix in image file
def create_confmatrix():
	expected = pd.read_csv(TEST_CONF_CSV).sort_values(by=["fname"])["label"].tolist()
	predicted = pd.read_csv(OUTPUT_CSV).sort_values(by=["fname"])["label"].tolist()
	labels = pd.read_csv(TEST_CONF_CSV).sort_values(by=["label"])["label"].drop_duplicates().tolist()
	results = confusion_matrix(expected, predicted,labels)

	df_cm = pd.DataFrame(results, index = [i for i in labels],columns = [i for i in labels])
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True)
	plt.savefig('../final/confusion_matrix.png', format='png')


def main():
	utils.write_log_msg("Run MNN code ...")
	#limitedCsv()
	run_mnn(0)
	#create_confmatrix()

main()

