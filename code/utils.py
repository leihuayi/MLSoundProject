#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import pandas as pd
import os

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   create_distionary()                                                                         #
#                                                                                               #
#   Description:                                                                                #
#   Creates a dictionary of labels from a .csv file to be used for training.                    #
#                                                                                               #
#***********************************************************************************************#
def create_dictionary(file_name):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), file_name))
    labelList = df.groupby(["label"]).count().index.get_level_values("label").tolist()
    return {label: i for i, label in enumerate(labelList)}

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   print_csv_file()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Print the predict results along with filenames to the output csv file.                      #
#                                                                                               #
#***********************************************************************************************#
def print_csv_file(predicts, name_list, dictionary, output_path):
    file_ = open(output_path, "w")
    file_.write("fname,labels\n")
    for i, value in enumerate(predicts):
        file_.write("%s,%s\n" % (name_list[i], [k for k, v in dictionary.items() if v == value][0]))
