#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#



#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   create_distionary()                                                                         #
#                                                                                               #
#   Description:                                                                                #
#   Creates a dictionary of labels from a .csv file to be used for training.                    #
#                                                                                               #
#***********************************************************************************************#
def create_dictionary(file_name, column):
    # create an empty dictionary
    dictionary = {}
    
    # open file from the filename provided
    _file = open(file_name, "r")
    label = 0
    
    # check fro unique labels and generate a dictionary.
    for line in _file:
        split_data = line.split(",")
        if not (split_data[column] in dictionary):
            dictionary[split_data[column]] = label;
            label += 1
    
    # return the created dictionary to the calling program
    return dictionary

