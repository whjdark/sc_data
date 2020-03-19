import argparse
import os
import os.path as path
import sys

import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D, Dense,
                                     DepthwiseConv2D, Dropout, Flatten, Input,
                                     MaxPooling2D, ReLU, SeparableConv2D, add)
from tensorflow.keras.models import Model, Sequential, load_model
import autokeras as ak


def LoadAndSelectData(ad_batch_size):
    #cmd = 'python ./data_convet_seekfree.py -b %d' % ad_batch_size
    # print(cmd)
    #os.popen(cmd + ' > null.log').read()
    train_data = np.load('./ad_train_dat.npy')
    test_data = np.load('./ad_test_dat.npy')

    train_label = np.load('./pwm_train_label.npy')
    test_label = np.load('./pwm_test_label.npy')
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    ad_size = 1
    # Load and select data
    x_train, x_test, y_train, y_test = LoadAndSelectData(ad_size)

    x_train = x_train.reshape(int(x_train.size / ad_size/7), ad_size, 7, 1)
    x_test = x_test.reshape(int(x_test.size / ad_size/7), ad_size, 7, 1)
    print((x_train.shape, y_train.shape))
    print((x_test.shape, y_test.shape))

    x_train = x_train.astype('int8')
    y_train = y_train.astype('int8')
    x_test = x_train.astype('int8')
    y_test = y_train.astype('int8')
    #print('Training data shape:%d' % (min(x_train.flatten())))
    x_train = (x_train / 128).astype('float32')
    x_test = (x_test / 128).astype('float32')
    y_train = ((y_train)/128).astype('float32')
    y_test = ((y_test) / 128).astype('float32')

    # Initialize the classifier.
    reg = ak.ImageRegressor(max_trials=20)
    # x is the path to the csv file. y is the column name of the column to predict.
    reg.fit(x=x_train, y=y_train, epochs=30)
    # Evaluate the accuracy of the found model.
    print(reg.evaluate(x=x_test, y=y_test))

    model = reg.export_model()

    model.save("autokeras")
