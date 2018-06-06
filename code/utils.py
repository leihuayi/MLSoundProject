#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import os
import time
import pandas as pd
import numpy as np
from threading import Thread

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__),"../data/status.txt")

#***********************************************************************************************#
#                                                                                               #
#   Class:                                                                                      #
#   ThreadWithReturnValue                                                                       #
#                                                                                               #
#   Description:                                                                                #
#   Runs program in threads to speed up computation                                             #
#                                                                                               #
#***********************************************************************************************#
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        Thread.join(self)
        return self._return

#***********************************************************************************************#
#                                                                                               #
#   Class:                                                                                      #
#   Config                                                                                      #
#                                                                                               #
#   Description:                                                                                #
#   Class for storing learning parameters that are shared between different functions.          #
#                                                                                               #
#***********************************************************************************************#
class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)

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
    file_.write("fname,label\n")
    for i, value in enumerate(predicts):
        lbl_1 = [k for k, v in dictionary.items() if v == value[0]][0]
        lbl_2 = [k for k, v in dictionary.items() if v == value[1]][0]
        lbl_3 = [k for k, v in dictionary.items() if v == value[2]][0]
        file_.write("%s,%s %s %s\n" % (name_list[i], lbl_1, lbl_2, lbl_3))
    # corrupt data append
    file_.write("0b0427e2.wav,Harmonica\n6ea0099f.wav,Harmonica\nb39975f5.wav,Harmonica")     

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   write_log_msg()                                                                             #
#                                                                                               #
#   Description:                                                                                #
#   Write status message to log file.                                                           #
#                (keep the console clean - console protection association :D)                   #
#                                                                                               #
#***********************************************************************************************#
def write_log_msg(message="", log_path = DEFAULT_LOG_PATH, newline = True):
    # check if newline feed needs to be added
    if newline:
        message = message + "\n"
    # write the log message to the file
    with open(log_path, 'a') as log_file:
        log_file.write(message)

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   initialize_log()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Prepare the log file by printing header with run info.                                      #
#                                                                                               #
#***********************************************************************************************#
def initialize_log(log_path = DEFAULT_LOG_PATH):
    # prepare header for the log file
    header =    "\n\n\n##################################################################################\n"
    header = header + "#    RUN DATE: " + time.strftime("%d/%m/%Y") + "                                  RUN TIME: " + time.strftime("%H:%M:%S") + "    #\n"
    header = header + "##################################################################################\n\n"
    # write the log message to the file
    with open(log_path, 'a') as log_file:
        log_file.write(header)

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   generate_chunks()                                                                           #
#                                                                                               #
#   Description:                                                                                #
#   Splits the given list into evenly sized chunks of provided size.                            #
#                                                                                               #
#***********************************************************************************************#
def generate_chunks(input_list, chunk_size):
    # For item i in a range that is a length of input_list,
    for i in range(0, len(input_list), chunk_size):
        # Create an index range for input_list of chunk_size items:
        yield input_list[i:i+chunk_size]
