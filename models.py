import tensorflow
#from tensorflow.keras import backend as K
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, ReLU
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, SeparableConv2D, SpatialDropout2D, add, GaussianNoise, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.initializers import glorot_uniform
import os
import sys


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = SeparableConv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                        name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = ReLU(max_value=6)(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = SeparableConv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                        name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = ReLU(max_value=6)(X)

    # Third component of main path (≈2 lines)
    X = SeparableConv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                        name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = add([X, X_shortcut])
    X = ReLU(max_value=6)(X)

    ### END CODE HERE ###

    return X


def MyResnet(input_shape=(1, 7, 1), output_shape=1):
    X_input = Input(shape=input_shape)
    X = X_input
    #X = ZeroPadding2D((3, 3))(X_input)

    X = SeparableConv2D(filters=64, kernel_size=(1, 7),
                        padding='same', strides=(2, 2))((X))
    X = BatchNormalization()(X)
    X = ReLU(max_value=6)(X)
    X = MaxPooling2D(1, 3)(X)

    X = identity_block(X, 3, [64, 64, 64], 1, "conv1")

    X = SeparableConv2D(filters=128, kernel_size=(1, 3),
                        padding='same', strides=(1, 1))((X))
    X = BatchNormalization()(X)
    X = ReLU(max_value=6)(X)

    X = identity_block(X, 3, [128, 128, 128], 1, "conv2")

    X = SeparableConv2D(filters=256, kernel_size=(1, 3),
                        padding='same', strides=(1, 1))((X))
    X = BatchNormalization()(X)
    X = ReLU(max_value=6)(X)

    X = identity_block(X, 3, [256, 256, 256], 1, "conv3")

    # X = SeparableConv2D(filters=512, kernel_size=(1, 3),
    #                    padding='same', strides=(1, 1))((X))
    #X = BatchNormalization()(X)
    #X = ReLU(max_value=6)(X)

    #X = identity_block(X, 3, [512, 512, 512], 1, "conv4")

    #X = AveragePooling2D(pool_size=(1, 7))(X)
    X = GlobalAveragePooling2D()(X)
    # -----------------
    X = Flatten()(X)
    # -----------
    X = Dense(output_shape)(X)
    X = BatchNormalization()(X)

    model = Model(inputs=X_input, outputs=X, name='MyResnet')

    return model


def CreateModelCNN5x3(num_classes=1, drop=0.25, isBN=True, ad_batch_size=10):
    model = MyResnet()
    model.summary()
    sModelName = 'smartcar_ad_cnn5x3_drop_0%d' % (int(drop*100))
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model

# CreateModelCNN5x3()


def CreateModelDense(num_classes=1, drop=0.25, isBN=True, ad_batch_size=1):
    model = Sequential()
    # -----------------
    model.add(Flatten(input_shape=(ad_batch_size, 7, 1)))

    model.add(Dense(140))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation("tanh"))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(140))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation("tanh"))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(70))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation("tanh"))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    model.summary()
    sModelName = 'smartcar_ad_dense_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model


def CreateModelHugeDense(num_classes=1, drop=0.25, isBN=True, ad_batch_size=1):
    model = Sequential()
    # -----------------
    model.add(Flatten(input_shape=(ad_batch_size, 7, 1)))

    model.add(Dense(1024))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(512))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(128))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    model.summary()
    sModelName = 'smartcar_ad_hugedense_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model


def CreateModelBaseline(num_classes=1, drop=0.25, isBN=True, ad_batch_size=1):
    # -----------------
    isBN = 0
    model = Sequential()
    # -----------------
    model.add(Flatten(input_shape=(ad_batch_size, 7, 1)))

    model.add(Dense(140))
    model.add(Activation('tanh'))
    if isBN:
        model.add(BatchNormalization())
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(100))
    model.add(Activation('tanh'))
    if isBN:
        model.add(BatchNormalization())
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(40))
    model.add(ReLU(max_value=8))
    if isBN:
        model.add(BatchNormalization())
    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    model.summary()
    sModelName = 'smartcar_ad_Baseline_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model


def CreateRNNModle(num_classes=1, drop=0.25, isBN=True, ad_batch_size=1):
    model = Sequential()

    model.add(tensorflow.keras.layers.GaussianNoise(
        stddev=0.01, input_shape=(ad_batch_size, 7, 1)))

    model.add(SeparableConv2D(filters=16, kernel_size=(7, 7),
                              padding='same', strides=(2, 2),
                              input_shape=(ad_batch_size, 7, 1)))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=32, kernel_size=(7, 7),
                              padding='same', strides=(2, 2)))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=64, kernel_size=(7, 7),
                              padding='same', strides=(2, 2)))
    if isBN:
        model.add(BatchNormalization())
    # -----------------
    model.add(Flatten())
    model.add(tensorflow.keras.layers.Reshape(target_shape=(8, 8)))
    model.add(tensorflow.keras.layers.Bidirectional(
        tensorflow.keras.layers.GRU(64, return_sequences=True)))
    model.add(tensorflow.keras.layers.Bidirectional(
        tensorflow.keras.layers.GRU(32)))

    model.add(Dense(16))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    if drop > 0:
        model.add(tensorflow.keras.layers.GaussianDropout(drop))

    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    model.summary()
    sModelName = 'smartcar_ad_RNN_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model

# CreateRNNModle()


def CreateModelCNN(num_classes=1, drop=0.25, isBN=True, ad_batch_size=10):
    model = Sequential()

    model.add(SeparableConv2D(filters=32, kernel_size=(7, 7),
                              padding='same', strides=(2, 2),
                              input_shape=(ad_batch_size, 7, 1)))
    model.add(ReLU(max_value=8))
    if isBN:
        model.add(BatchNormalization())

    model.add(SeparableConv2D(filters=32, kernel_size=(5, 5),
                              padding='same', strides=(2, 2)))
    model.add(ReLU(max_value=8))
    if isBN:
        model.add(BatchNormalization())

    model.add(SeparableConv2D(filters=48, kernel_size=(3, 3),
                              padding='same', strides=(2, 2)))
    model.add(ReLU(max_value=8))
    if isBN:
        model.add(BatchNormalization())

    model.add(SeparableConv2D(filters=64, kernel_size=(3, 3),
                              padding='same', strides=(1, 1)))
    model.add(ReLU(max_value=8))
    if isBN:
        model.add(BatchNormalization())

    model.add(SeparableConv2D(filters=64, kernel_size=(1, 1),
                              padding='same', strides=(1, 1)))
    model.add(ReLU(max_value=8))
    if isBN:
        model.add(BatchNormalization())

    # -----------------
    model.add(Flatten())

    # -----------
    model.add(Dense(64))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(48))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    model.summary()
    sModelName = 'smartcar_ad_cnn7x5_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model


def CreateModelconv1d(num_classes=1, drop=0.25, isBN=True, ad_batch_size=10):
    model = Sequential()
    pass

    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    #model.build(input_shape=[None, 1, 7])
    # model.summary()
    sModelName = 'smartcar_ad_conv1d_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model


def CreateModelCNNV2(num_classes=1, drop=0.25, isBN=True, ad_batch_size=1):
    model = Sequential()
    model.add(SeparableConv2D(filters=14, kernel_size=(7, 7),
                              padding='same', strides=(2, 2),
                              input_shape=(ad_batch_size, 7, 1)))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=28, kernel_size=(3, 3),
                              padding='same', strides=(2, 2)))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=56, kernel_size=(3, 3),
                              padding='same', strides=(2, 2)))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=56, kernel_size=(3, 3),
                              padding='same', strides=(1, 1)))
    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    # -----------------
    model.add(Flatten())
    # -----------
    model.add(Dense(56))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(28))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(28))
    if isBN:
        model.add(BatchNormalization())
    model.add(Activation('tanh'))
    if drop > 0:
        model.add(Dropout(drop))

    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    model.summary()
    sModelName = 'smartcar_ad_CNNV2_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model

# CreateModelCNNV2()


def smallpureCNNModel(num_classes=1, drop=0.25, isBN=True, ad_batch_size=1):

    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(1, 3),
                     padding='same', strides=(1, 1),
                     input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=32, kernel_size=(1, 3),
                     padding='same', strides=(1, 1),
                     input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))
    # model.add(Dropout(drop))

    model.add(SeparableConv2D(filters=64, kernel_size=(1, 3),
                     padding='same', strides=(1, 1),
                     input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=128, kernel_size=(1, 3),
                     padding='same', strides=(1, 1),
                     input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))
    # model.add(Dropout(drop))

    model.add(GlobalAveragePooling2D())

    model.add(Flatten())

    #model.add(Dense(64))
    #if isBN:
    #    model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    #model.add(Dropout(drop))

    model.add(Dense(num_classes))
    if isBN:
        model.add(BatchNormalization())

    model.summary()
    sModelName = 'smartcar_ad_pureCNN_drop_0%d_adSize_%d' % (
        int(drop * 100), ad_batch_size)
    if not isBN:
        sModelName += '_nobn'
    return sModelName, model


def smallpureCNNModelV2(num_classes=1, drop=0.25, isBN=True, ad_batch_size=1):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(1, 7),
                     padding='same', strides=(1, 1),
                     input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=64, kernel_size=(1, 7),
                              padding='same', strides=(1, 1),
                              input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))
    # model.add(Dropout(drop))

    model.add(SeparableConv2D(filters=64, kernel_size=(1, 7),
                              padding='same', strides=(1, 1),
                              input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))

    model.add(SeparableConv2D(filters=512, kernel_size=(1, 7),
                              padding='same', strides=(1, 1),
                              input_shape=(ad_batch_size, 7, 1)))

    if isBN:
        model.add(BatchNormalization())
    model.add(ReLU(max_value=8))
    # model.add(Dropout(drop))

    model.add(GlobalAveragePooling2D())

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

# smallpureCNNModel()
