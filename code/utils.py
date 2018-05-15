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

