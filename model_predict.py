import numpy as np
import tensorflow as tf
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
import math

def calc_diff_calculus(input1,input2):
    out_arr = []
    tmp_arr = input1 - input2
    for i in range(input1.size):
        if (i>0):
            out_arr.append(out_arr[i-1] + (input1[i] - input2[i]))
        else:
            out_arr.append(input1[i] - input2[i])

    return np.array(out_arr);

if __name__ == "__main__":
    x_train = np.load('./ad_train_dat.npy')
    y_train = np.load('./pwm_train_label.npy')
    x_test = np.load('./ad_test_dat.npy')
    y_test = np.load('./pwm_test_label.npy')


    x_test = x_test.astype('int16')
    #print(max(x_test.flatten()))
    #print(min(x_test.flatten()))
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2],1)
    y_test = y_test.reshape(y_test.size,1)
    y_train = y_train.reshape(y_train.size, 1)
    
    #mode_name = 'HugeDense/smartcar_ad_hugedense_drop_025_adSize_1.h5'
    #mode_name = 'forevergod/ak_cnn.h5'
    mode_name = 'smartcar_ad_pureCNN_drop_025_adSize_1.h5'
    #mode_name = 'trials/dense(dropout0.5 relu8)/smartcar_ad_dense_drop_050_adSize_1.h5'
    model = load_model(mode_name)

    x_input = x_test
    y_input = y_test

    x_input = x_input / 128.0
    y_input = y_input / 128.0


    #print(max(y_input.flatten()))
    #print(min(y_input.flatten()))

    print('x_test data shape:(%f~%f)' % (max(x_input.flatten()), min(x_input.flatten())))

    x_input = x_input.astype('float32')
    y_input = y_input.astype('float32')
    y_out = model.predict(x_input)


    Index = model.evaluate(x_input, y_input)
    print("evaluate: ",Index)
    y_input = y_input.reshape(y_input.size,1)

    y_input = (y_input) * 128.0
    y_out = (y_out) * 128.0

    y_diff_out = calc_diff_calculus(y_out,y_input) #error  accumulation
    print('error accumulation : ', tf.reduce_sum(y_diff_out))

    x = np.arange(0,y_input.size)

    plt.plot(x,y_input, label='original pwm')
    plt.plot(x,y_out, color='r',label='predict pwm')
    plt.plot(x, y_diff_out, color='g', label='diff')
    plt.title('Blue:original pwm\nRed:predict pwm')
    plt.show()
    #print(y_out)
    #print(y_input)

    #loss = tf.losses.mean_squared_error(y_input,y_out)
    #with tf.Session() as sess:
    #    print("predict: ",sess.run(loss))

    mse = tf.reduce_mean(tf.square(y_input - y_out))
    print("R : %f--->%f-->%f" %(mse, math.sqrt(mse), math.sqrt(mse)/128))

    print("end")


