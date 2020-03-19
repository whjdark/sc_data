import tensorflow as tf
import numpy as np
from tensorflow import keras
 
#加载模型
keras_file = 'simpleCNN/smartcar_ad_cnn7x5_drop_025_adSize_1.h5'
model = tf.keras.models.load_model(keras_file)

# 转换模型。
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 加载 TFLite 模型并分配张量（tensor）。
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 使用随机数据作为输入测试 TensorFlow Lite 模型。
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32) / 128.0
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# 函数 `get_tensor()` 会返回一份张量的拷贝。
# 使用 `tensor()` 获取指向张量的指针。
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# 使用随机数据作为输入测试 TensorFlow 模型。
tf_results = model(tf.constant(input_data))

# 对比结果。
for tf_result, tflite_result in zip(tf_results * 128.0, tflite_results * 128.0):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=4)

if input("save? ") == "yes":
    open("lite_models/simpleCNN.tflite", "wb").write(tflite_model)
