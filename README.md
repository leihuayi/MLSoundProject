Machine Learning for Sound Classification, Tsinghua University.
Project developped by Usama and Sarah in Python.

# Project overview

This aim of this project is to correctly classify environmental sounds. These can be from various natures, from human laughter to cello or glass shatter.
After performing feature extraction (namely MFCC) from the Freesoudns library dataset using librosa, we developped several algorithms based on neural networks.
* Multilayer neural network (56,4% accuracy)
* Convolutional neural network with 2 Dimentions (84,2%)
* Convolutional neural network with 1 Dimention (89,6% combined with 2D)


# Project evaluation

This project concurred for a kaggle competition https://www.kaggle.com/c/freesound-audio-tagging. We were finally ranked as 158/558.
After incremental trials, we went from a score of 0.2 to 0.90, measured by the Mean Average Precision @ 3. 



