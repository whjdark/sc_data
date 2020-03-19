import tensorflow as tf
import tensorflow.keras.models 

def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops

#get_flops("./rnn/smartcar_ad_RNN_drop_025_adSize_1.h5")
#get_flops("./simpleCNN/smartcar_ad_cnn7x5_drop_025_adSize_1.h5")
#get_flops("./simpleCNNV2.1/smartcar_ad_CNNV2_drop_025_adSize_1.h5")
#get_flops("./bigdense/smartcar_ad_dense_drop_025_adSize_1.h5")
get_flops("smartcar_ad_pureCNN_drop_025_adSize_1.h5")
#get_flops("./forevergod/ak_cnn.h5")
#model = tensorflow.keras.models.load_model("./simpleCNNV2.1/smartcar_ad_CNNV2_drop_025_adSize_1.h5")
#model.summary()
