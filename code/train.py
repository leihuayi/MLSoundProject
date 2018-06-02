#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import numpy as np
import pandas as pd
import utils
import tensorflow as tf

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   train()                                                                                     #
#                                                                                               #
#   Description:                                                                                #
#   The training module of the project. Responsible for training the parameters for provided    #
#   features and selected options.                                                              #
#                                                                                               #
#***********************************************************************************************#
def train(tr_features, tr_labels, ts_features, n_classes, training_epochs = 5000):
     mnn_y_pred, mnn_probs, mnn_pred = tensor_multilayer_neural_network(tr_features, tr_labels, ts_features, n_classes, training_epochs)
     return ensemble_results(mnn_probs, mnn_pred)
    
#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   ensemble_results()                                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Ensemble the results of all the models and return top 3 predictions.                        #
#                                                                                               #
#***********************************************************************************************#
def ensemble_results(mnn_probs, mnn_pred):
    top3 = mnn_pred[:, [0, 1, 2]]
    return top3

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   tensor_multilayer_neural_network()                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Using tensorflow library to build a simple multi layer neural network.                      #
#                                                                                               #
#***********************************************************************************************#
def tensor_multilayer_neural_network(tr_features, tr_labels, ts_features, n_classes, training_epochs):
    # initialize the beginning paramters.
    n_dim = tr_features.shape[1]
    n_hidden_units_1 =  200   #280 
    n_hidden_units_2 =  250  #300
    n_hidden_units_3 =  300  #300
    
    sd = 1 / np.sqrt(n_dim)
    
    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,n_classes])

    # initializing starting learning rate - will use decaying technique
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.005, global_step, 500, 0.95, staircase=True)
    
    # initialize layer 1 parameters
    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_1], mean = 0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_1], mean = 0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

    # initialize layer 2 parameters
    W_2 = tf.Variable(tf.random_normal([n_hidden_units_1,n_hidden_units_2], mean = 0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_2], mean = 0, stddev=sd))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

    # initialize layer 3 parameters
    W_3 = tf.Variable(tf.random_normal([n_hidden_units_2,n_hidden_units_3], mean = 0, stddev=sd))
    b_3 = tf.Variable(tf.random_normal([n_hidden_units_3], mean = 0, stddev=sd))
    h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3)
    
    W = tf.Variable(tf.random_normal([n_hidden_units_3,n_classes], mean = 0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
    y_ = tf.nn.softmax(tf.matmul(h_3,W) + b)

    cost_function = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function, global_step=global_step) #GradientDescentOptimizer(learning_rate).minimize(cost_function, global_step=global_step)

    init = tf.global_variables_initializer()
    
    cost_history = np.empty(shape=[1],dtype=float)
    y_pred = None
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):            
            # print a log message for status update
            utils.write_log_msg("running the training epoch {0}...".format(epoch+1))
            # running the training_epoch numbered epoch
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
            cost_history = np.append(cost_history,cost)
        # predict results based on the trained model
        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
        y_k_probs, y_k_pred = sess.run(tf.nn.top_k(y_, k=n_classes), feed_dict={X: ts_features})
    # plot cost history
    df = pd.DataFrame(cost_history)
    df.to_csv("../data/cost_history.csv")

    # return the predicted values back to the calling program
    return y_pred, y_k_probs, y_k_pred

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   convolution_1D()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Building a 1 dimentional convolutional network for training and prediction of audio tags.   #
#                                                                                               #
#***********************************************************************************************#
def convolution_1D(tr_features, tr_labels, ts_features, n_classes, training_epochs):
    print("Please build the model")
    
#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   convolution_2D()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Building a 2 dimentional convolutional network for training and prediction of audio tags.   #
#                                                                                               #
#***********************************************************************************************#
def convolution_2D(tr_features, tr_labels, ts_features, n_classes, training_epochs):
    print("Please build the model")
