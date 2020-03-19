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

import models

#import data_convert



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
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restart',
                        help='restart training', action='store_true')
    parser.add_argument(
        '-n', '--no_bn', help='restart training', action='store_true')
    parser.add_argument(
        '-d', '--drop', type=float, default=0.25,
        help="""\
        dropout rate, 0 - 1
        """)
    parser.add_argument(
        '-l', '--learn_rate', type=float, default=0.01,
        help="""\
        learning rate
        """)
    parser.add_argument(
        '-mlr', '--min_learn_rate', type=float, default=0.0007,
        help="""\
        min learning rate, will not decay when learning rate is less than this
        """)

    parser.add_argument(
        '-c', '--decay_ppm', type=float, default=1.5,
        help="""\
        learning rate decay rate, in ppm: 1ppm = 1e-6
        """)
    parser.add_argument(
        '-b', '--batch_size', type=int, default=64,
        help="""\
        batch size
        """)
    parser.add_argument(
        '-ad', '--ad_size', type=int, default=1,
        help="""\
            batch size
            """)
    parser.add_argument(
        '-a', '--arch', type=str, default='dense',
        help="""\
        model architecture: can be 'ds', 'cnn', 'resds', 'cnnfast'
        """)
    parser.add_argument(
        '-E', '--EPOCH', type=int, default=120,
        help=""
    )
    parser.add_argument(
        '-e', '--epochpertrain', type=int, default=1,
        help=""
    )

    args, unparsed = parser.parse_known_args()

    ad_size = args.ad_size
    # Load and select data
    x_train, x_test, y_train, y_test = LoadAndSelectData(ad_size)

    lr = args.learn_rate
    minLR = args.min_learn_rate
    batch_size = args.batch_size
    decay = args.decay_ppm / 1e6
    histCnt = 20
    lossHist = [1E4] * histCnt
    epocsPerTrain = args.epochpertrain
    burstOftCnt = 0
    minLoss = 1E5
    minmae = 1E5
    minmse = 1E5
    if args.arch == 'RNN' :
        opt = tensorflow.keras.optimizers.RMSprop(lr= 0.001)
    else:
        opt = tensorflow.keras.optimizers.Nadam(lr, decay)
    #opt = tensorflow.keras.optimizers.SGD(lr, momentum=0.0, nesterov=False)
    num_classes = 1

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
    # print('x_test data shape:%f~%f' %
    #      (max(x_train.flatten()), min(x_train.flatten())))
    #print('y_test data shape:%f~%f' % (max(y_train), min(y_train)))

    if(input("NEW?") == "yes"):
        if args.arch == '7x5':
            model_name, model = models.CreateModelCNN(
                num_classes, args.drop, not args.no_bn, ad_size)
        elif args.arch == 'huge':
            model_name, model = models.CreateModelHugeDense(
                num_classes, args.drop, not args.no_bn, ad_size)
        elif args.arch == 'CNNV2':
            model_name, model = models.CreateModelCNNV2(
                num_classes, args.drop, not args.no_bn, ad_size)
        elif args.arch == '5x3':
            model_name, model = models.CreateModelCNN5x3(
                num_classes, args.drop, not args.no_bn, ad_size)
        elif args.arch == '1d':
            model_name, model = models.CreateModelconv1d(
                num_classes, args.drop, not args.no_bn, ad_size)
        elif args.arch == 'dense':
            model_name, model = models.CreateModelDense(
                num_classes, args.drop, not args.no_bn, ad_size)
        elif args.arch == 'RNN':
            model_name, model = models.CreateRNNModle(
                num_classes, args.drop, not args.no_bn, ad_size)
        elif args.arch == 'baseline':
            model_name, model = models.CreateModelBaseline(
                num_classes, args.drop, not args.no_bn, ad_size)
        elif args.arch == 'pure':
            model_name, model = models.smallpureCNNModel(
                num_classes, args.drop, not args.no_bn, ad_size)
        else:  # default
            model_name, model = models.CreateModelBaseline(
                num_classes, args.drop, not args.no_bn, ad_size)
    else:
        model_name = "smartcar_ad_cnn7x5_drop_025_adSize_1"
        sSaveCtx = '%s_ctx.h5' % (model_name)
        model = load_model(sSaveCtx)
        
    model.summary()

    if input("train?") == "no":
        print("EXIT")
        exit()
    #model_name, model = CreateModelDense(num_classes, args.drop, not args.no_bn, ad_size)
    print('Training model ' + model_name)
    # train the model using ADAMAX
    logFile = model_name + '_log.txt'
    sSaveModel = '%s.h5' % (model_name)
    sSaveCtx = '%s_ctx.h5' % (model_name)
    if args.restart == False and path.exists(sSaveCtx):
        fd = open(logFile, 'r')
        s = fd.read()
        fd.close()
        lst = s.split('\n')[-2].split(',')
        for s in lst:
            if s.find('lr=') == 0:
                lr = float(s[3:])
                lr *= (1 - decay) ** (50000 / batch_size)
            if s.find('times=') == 0:
                i = int(s[6:]) - 1 + 1
            if s.find('loss=') == 0:
                minLoss = float(s[5:])
            if s.find('mae=') == 0:
                minmae = float(s[4:])
            if s.find('mse=') == 0:
                minmse = float(s[4:])
        print('resume training from ', lst)
        model = load_model(sSaveCtx)
        fd = open(logFile, 'a')
    else:
        fd = open(logFile, 'w')
        s = 'times=%d,loss=%.4f,mae=%.6f,mse=%.6f,lr=%f,decay=%f' % (
            0, minmae, minmse, minLoss, lr, decay)
        fd.write(s + '\n')
        fd.close()
        fd = open(logFile, 'a')
        i = 0
    model.compile(loss='mean_squared_error',
                  optimizer=opt, metrics=['mae', 'mse'])

    while i < args.EPOCH:
        print('Train %d times' % (i + 1))
        hist = model.fit(x_train, y_train, batch_size, epochs=epocsPerTrain,
                         shuffle=True, callbacks=None)  # callbacks=[TensorBoard(log_dir='./log_' + model_name)])
        model.save(sSaveCtx)
        # evaluate
        loss, mae, mse = model.evaluate(x_test, y_test)
        # process loss
        loss = int(loss * 10000) / 10000.0

        if loss < minLoss:
            minLoss = loss
            print('Saved a better model!')
            model.save(sSaveModel)
            s = 'Saved times=%d,loss=%.4f,lr=%f,decay=%f,mae=%f,mse=%f' % (
                i+1, minLoss, lr, decay, mae, mse)
        else:
            # save log
            s = 'times=%d,loss=%.4f,lr=%f,decay=%f,mae=%f,mse=%f' % (
                i+1, minLoss, lr, decay, mae, mse)
        # check if it is overfit
        oftCnt = 0
        for k in range(histCnt):
            if loss > lossHist[k]:
                oftCnt += 1
        oftRate = oftCnt / histCnt
        print(',overfit rate = %d%%' % int(oftRate * 100))
        if oftCnt / histCnt >= 0.6:
            burstOftCnt += 1
            if burstOftCnt > 3:
                print('Overfit!')
        else:
            burstOftCnt = 0
        s = s + ',overfit rate = %d%%' % int(oftRate * 100)

        lossHistPrnCnt = 6
        if lossHistPrnCnt > histCnt:
            lossHistPrnCnt = histCnt
        fd.write(s + '\n')
        print(s, lossHist[:lossHistPrnCnt])
        fd.close()

        # update loss history
        for k in range(histCnt - 1):
            ndx = histCnt - 1 - k
            lossHist[ndx] = lossHist[ndx - 1]
        lossHist[0] = loss

        fd = open(logFile, 'a')
        lr *= (1 - decay) ** (50000 / batch_size)
        if lr < minLR:
            lr = minLR
        model = load_model(sSaveCtx)
        if args.arch == 'RNN' :
            opt = tensorflow.keras.optimizers.RMSprop(lr= 0.001)
        else:
            opt = tensorflow.keras.optimizers.Nadam(lr, decay)
        #opt = tensorflow.keras.optimizers.SGD(lr, momentum=0.0, nesterov=False)
        print('new lr_rate=%f' % (lr))
        model.compile(loss='mean_squared_error',
                      optimizer=opt, metrics=['mae', 'mse'])
        i += 1
