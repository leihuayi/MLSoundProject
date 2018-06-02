#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------# 
import numpy as np
import pandas as pd
import utils
import tensorflow as tf

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
frames = 41
bands = 60

feature_size = 2460 #60x41
num_labels = 10
num_channels = 2

batch_size = 50
kernel_size = 30
depth = 20
num_hidden = 200

learning_rate = 0.01
total_iterations = 2000

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
    # call the multi-layer neural network to get results
    mnn_y_pred, mnn_probs, mnn_pred = tensor_multilayer_neural_network(tr_features, tr_labels, ts_features, n_classes, training_epochs)
    # call the 1d convolutional network code here
    
    # call the 2d convolutional network code here
    cnn_2d_y_pred, cnn_2d_probs, cnn_2d_pred = tensor_convolution_2D(tr_features, tr_labels, ts_features, n_classes, training_epochs)
    
    # ensemble the results to get combined prediction
    return ensemble_results(mnn_probs, mnn_pred, cnn_2d_probs, cnn_2d_pred)
    
#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   ensemble_results()                                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Ensemble the results of all the models and return top 3 predictions.                        #
#                                                                                               #
#***********************************************************************************************#
def ensemble_results(mnn_probs, mnn_pred, cnn_2d_probs, cnn_2d_pred):
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
    df.to_csv("../data/cost_history_mnn.csv")

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
def convolution_1D(tr_features, tr_labels, ts_features, n_classes, training_epochs=2000):

    # CNN parameters
    feature_size = 2460 #60x41
    num_labels = 10
    num_channels = 2

    batch_size = 50
    kernel_size = 30
    depth = 20
    num_hidden = 200

    learning_rate = 0.01

    # Create CNN model
    X = tf.placeholder(tf.float32, shape=[None,feature_size,num_channels])
    Y = tf.placeholder(tf.float32, shape=[None,num_labels])

    cov1d = tf.nn.conv1d(X,weights,strides=[1,2,2,1], padding='VALID')
    cov = apply_convolution(kernel_size,num_channels,depth, cov1d)

    shape = cov.get_shape().as_list()
    cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

    f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
    f_biases = bias_variable([num_hidden])
    f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

    out_weights = weight_variable([num_hidden, num_labels])
    out_biases = bias_variable([num_labels])
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

    loss = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train CNN
    cost_history = np.empty(shape=[1],dtype=float)
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for itr in range(training_epochs):    
            offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
            batch_x = tr_features[offset:(offset + batch_size), :, :, :]
            batch_y = tr_labels[offset:(offset + batch_size), :]
            
            _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,c)

        # predict results based on the trained model
        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
        y_k_probs, y_k_pred = sess.run(tf.nn.top_k(y_, k=n_classes), feed_dict={X: ts_features})
        
    # plot cost history
    df = pd.DataFrame(cost_history)
    df.to_csv("../data/cost_history_cnn_1d.csv")

    # return the predicted values back to the calling program
    return y_pred, y_k_probs, y_k_pred


#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   convolution_2D()                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Building a 2 dimentional convolutional network for training and prediction of audio tags.   #
#                                                                                               #
#***********************************************************************************************#
def tensor_convolution_2D(tr_features, tr_labels, ts_features, n_classes, training_epochs):
    # CNN parameters
    frames = 41
    bands = 60

    feature_size = 2460 #60x41
    num_labels = 10
    num_channels = 2

    batch_size = 50
    kernel_size = 30
    depth = 20
    num_hidden = 200

    learning_rate = 0.01
    total_iterations = 2000

    # initial values declarations
    X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
    Y = tf.placeholder(tf.float32, shape=[None,num_labels])
    
    # building a convolutional network
    cov2d = tf.nn.conv2d(X,weights,strides=[1,2,2,1], padding='SAME')
    cov = apply_convolution(kernel_size,num_channels,depth,cov2d)

    shape = cov.get_shape().as_list()
    cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])
    
    f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
    f_biases = bias_variable([num_hidden])
    f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))
    
    out_weights = weight_variable([num_hidden, num_labels])
    out_biases = bias_variable([num_labels])
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
    
    loss = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    cost_history = np.empty(shape=[1],dtype=float)
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for itr in range(total_iterations):    
            offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
            batch_x = tr_features[offset:(offset + batch_size), :, :, :]
            batch_y = tr_labels[offset:(offset + batch_size), :]
            
            _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,c)
    
        # predict results based on the trained model
        y_pred = session.run(tf.argmax(y_,1),feed_dict={X: ts_features})
        y_k_probs, y_k_pred = session.run(tf.nn.top_k(y_, k=n_classes), feed_dict={X: ts_features})

    # plot cost history
    df = pd.DataFrame(cost_history)
    df.to_csv("../data/cost_history_cnn_2d.csv")

    # return the predicted values back to the calling program
    return y_pred, y_k_probs, y_k_pred
        
#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   helper functions                                                                            #
#                                                                                               #
#   Description:                                                                                #
#   Helper functions for building a 2D convolutional network.                                   #
#                                                                                               #
#***********************************************************************************************#
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def apply_convolution(kernel_size,num_channels,depth, cov_function):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(cov_function,biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1], padding='SAME')

