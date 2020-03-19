from tflite_runtime.interpreter import Interpreter
import numpy as np
import math
import time
import matplotlib.pyplot as plt

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

    start = time.time()
    x_test = x_test.astype('int16')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
    y_test = y_test.reshape(y_test.size,1)
    
    x_input = x_test
    y_input = y_test

    x_input = x_input / 128
    y_input = y_input / 128

    x_input = x_input.astype('float32')
    y_input = y_input.astype('float32')
    
    model_file = 'lite_models/smalldense.tflite'
    interpreter = Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    #print(interpreter)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_predict = []

    for input_tensor in x_input:
        input_data = np.expand_dims(input_tensor, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        y_predict.append(results)
    
    y_predict = np.asarray(y_predict)

    y_predict = y_predict * 128.0
    y_input = y_input * 128.0
    
    #mse = np.mean(np.square(y_input - y_predict))
    #print("R : %f--->%f-->%f" %(mse, math.sqrt(mse), math.sqrt(mse)/128))

    end = time.time()
    print("Execution Time: ", end - start)

    #print(y_input.size)
    #print(y_predict.size)

    y_diff_out = calc_diff_calculus(y_predict,y_input) #error  accumulation
    #print('error accumulation : ', np.sum(y_diff_out))
    x_aixs = np.arange(0,y_input.size)
    plt.plot(x_aixs,y_input, label='original pwm')
    plt.plot(x_aixs,y_predict, color='r',label='predict pwm')
    plt.title('Blue:original pwm\nRed:predict pwm')
    plt.plot(x_aixs, y_diff_out, color='g', label='diff')
    plt.show()