import tensorflow
#from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, ReLU
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, AveragePooling2D, MaxPooling2D, SeparableConv2D, SpatialDropout2D, add, GaussianNoise
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization
import os
import sys


def smallpureCNNModel(num_classes=1, drop=0.25, isBN=True, ad_batch_size=1):
    model = Sequential()
    model.add(SeparableConv2D(filters=16, kernel_size=(7, 7),
                              padding='same', strides=(1, 1),
                              input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=64, kernel_size=(7, 7),
                              padding='same', strides=(1, 1),
                              input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))
    model.add(Dropout(drop))

    model.add(SeparableConv2D(filters=32, kernel_size=(7, 7),
                              padding='same', strides=(1, 1),
                              input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=512, kernel_size=(7, 7),
                              padding='same', strides=(1, 1),
                              input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))
    model.add(Dropout(drop))

    model.add(Flatten())

    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    model.summary()
    sModelName = 'smartcar_ad_pureCNN_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model