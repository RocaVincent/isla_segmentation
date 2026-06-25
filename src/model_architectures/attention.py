import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow_graphics.math.interpolation.trilinear import interpolate as trilinear_interpolate

class TrilinearUpsampling(layers.Layer):

    def __init__(self, factors):
        super(TrilinearUpsampling, self).__init__()
        self.factors = factors

    def build(self, input_shape):
        dimensions = [f*d for (f,d) in zip(self.factors,input_shape[1:-1])]
        self.indices = np.indices(dimensions)
        self.indices = np.moveaxis(self.indices, 0, -1) / self.factors
        self.output_shape_ = list(self.indices.shape[:-1])+[input_shape[-1]]
        self.indices = self.indices.reshape((-1,3))
        self.indices = tf.constant(self.indices, dtype=self.compute_dtype)

    def upsample_entry(self, entry):
        x = trilinear_interpolate(entry, self.indices)
        return tf.reshape(x, self.output_shape_)

    def call(self, x):
        return tf.map_fn(self.upsample_entry, x, fn_output_signature=tf.TensorSpec(shape=self.output_shape_, dtype=self.compute_dtype))
    

class SE_block(layers.Layer):
    
    def __init__(self, reduction_ratio):
        super(SE_block, self).__init__()
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.fc1 = layers.Dense(units=in_channels//self.reduction_ratio, use_bias=True, activation='relu')
        self.fc2 = layers.Dense(units=in_channels, use_bias=True)
        self.sigmoid = layers.Activation('sigmoid', dtype='float32')
        
    def call(self, x):
        inp = x if self.compute_dtype=='float32' else tf.cast(x, 'float32')
        sq = tf.math.reduce_mean(inp, axis=(1,2,3), keepdims=True)
        if self.compute_dtype != 'float32': sq = tf.cast(sq, self.compute_dtype)
        sq = self.fc1(sq)
        sq = self.fc2(sq)
        sq = self.sigmoid(sq)
        if self.compute_dtype != 'float32': sq = tf.cast(sq, self.compute_dtype)
        return x*sq  
    
    
class Attention_block(layers.Layer):
    
    def __init__(self, up_down_sample):
        super(Attention_block, self).__init__()
        self.up_down_sample = up_down_sample
        
    def build(self, input_shape):
        int_channels = input_shape[0][-1] // 2 # default value according to Oktay et al's code.
        self.conv_encode = layers.Conv3D(filters=int_channels, kernel_size=self.up_down_sample, strides=self.up_down_sample, use_bias=False, padding='same')
        self.conv_decode = layers.Conv3D(filters=int_channels, kernel_size=1, strides=1, use_bias=True, padding='same')
        self.scale_conv = layers.Conv3D(filters=1, kernel_size=1, strides=1, use_bias=True, padding='same')
        self.sigmoid = layers.Activation('sigmoid', dtype='float32')
        self.upsample = TrilinearUpsampling(self.up_down_sample)
        
    def call(self, inputs):
        encoded, decoded = inputs
        enc_fm = self.conv_encode(encoded)
        dec_fm = self.conv_decode(decoded)
        scale = self.scale_conv(tf.math.maximum(enc_fm+dec_fm, 0))
        scale = self.sigmoid(scale)
        if self.compute_dtype!='float32': scale = tf.cast(scale, dtype=self.compute_dtype)
        scale = self.upsample(scale)
        return scale*encoded