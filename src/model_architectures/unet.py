from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from functools import partial
from enum import Enum

from .attention import SE_block, Attention_block
    
class UNetType(Enum):
    StdUNet = 'StdUNet'
    ResUNet = 'ResUNet'
    
class InputShape(Enum):
    SHAPE1 = (160,192,80)
    SHAPE2 = (256,256,48)
    
class InputResol(Enum):
    RESOL1 = (1.0,1.0,2.0)
    RESOL2 = (0.8,0.8,3.0)

def StdUNet_block(n_filters:int, kernel_size:tuple[int,int,int], strides1:tuple[int,int,int], dropout:bool):
    layers_ = [
        layers.Conv3D(filters=n_filters, kernel_size=kernel_size, strides=strides1, padding='same', use_bias=False),
        layers.GroupNormalization(groups=n_filters),
        layers.LeakyReLU(0.01)
    ]
    if dropout: layers_.append(layers.SpatialDropout3D(0.2, data_format='channels_last'))
    layers_.extend([
        layers.Conv3D(filters=n_filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=False),
        layers.GroupNormalization(groups=n_filters),
        layers.LeakyReLU(0.01)
    ])
    if dropout: layers_.append(layers.SpatialDropout3D(0.2, data_format='channels_last'))
    return Sequential(layers_)


class ResUNet_block(layers.Layer):
    
    def __init__(self, first: bool, n_filters:int, kernel_size:tuple[int,int,int], strides1:tuple[int,int,int], dropout:bool):
        super().__init__()
        if first:
            self.block = Sequential([
                layers.Conv3D(filters=n_filters, kernel_size=kernel_size, strides=strides1, padding='same', use_bias=False),
                layers.GroupNormalization(groups=n_filters, gamma_initializer=tf.keras.initializers.Constant(0.1)),
                layers.ReLU(),
                layers.Conv3D(filters=n_filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=False),
            ])
        else:
            layers_ = [
                layers.GroupNormalization(groups=-1),
                layers.ReLU()
            ]
            if dropout: layers_.append(layers.SpatialDropout3D(0.2, data_format='channels_last'))
            layers_.extend([
                layers.Conv3D(filters=n_filters, kernel_size=kernel_size, strides=strides1, padding='same', use_bias=False),
                layers.GroupNormalization(groups=n_filters, gamma_initializer=tf.keras.initializers.Constant(0.1)),
                layers.ReLU()
            ])
            if dropout: layers_.append(layers.SpatialDropout3D(0.2, data_format='channels_last'))
            layers_.append(layers.Conv3D(filters=n_filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=False))
            self.block = Sequential(layers_)
        self.id_map = layers.Conv3D(filters=n_filters, kernel_size=1, strides=strides1, padding='same', use_bias=True)
        
    def call(self, inputs):
        return self.block(inputs) + self.id_map(inputs) 
    

def UNet(
    input_shape: InputShape,
    unet_type: UNetType,
    deep_supervision:bool,
    dropout_deepLayers:bool,
):   
    inputs = layers.Input(shape=list(input_shape.value)+[2])
    
    n_filters = 32,64,128,256,320,320
    if input_shape==InputShape.SHAPE1:
        kernel_size1 = 3,3,3
        downUp_spacing2 = 2,2,2
        downUp_spacing_last = 2,2,1
    else:
        kernel_size1 = 3,3,1
        downUp_spacing2 = 2,2,1
        downUp_spacing_last = 2,2,2
    kernel_sizes = [kernel_size1] + [(3,3,3)] * (len(n_filters)-1)
    downUp_spacings = [(1,1,1) , downUp_spacing2] + [(2,2,2)] * (len(n_filters)-3) + [downUp_spacing_last]
    dropouts = [False] * (len(n_filters)-3) + [dropout_deepLayers]*3
    use_deepSup = [deep_supervision] * (len(n_filters)-3) + [False]*2
    
    
    if unet_type == UNetType.ResUNet:
        x = ResUNet_block(first=True, n_filters=n_filters[0], kernel_size=kernel_sizes[0], strides1=downUp_spacings[0], dropout=dropouts[0])(inputs)
        skips = [x]
        Conv_block = partial(ResUNet_block, first=False)
        idx_conv = 1
    else:
        x = inputs
        skips = []
        Conv_block = StdUNet_block
        idx_conv = 0
    
    for i in range(idx_conv, len(n_filters)):
        x = Conv_block(n_filters=n_filters[i], kernel_size=kernel_sizes[i], strides1=downUp_spacings[i], dropout=dropouts[i])(x)
        if i != len(n_filters)-1:
            skip = SE_block(reduction_ratio=8)(x)
            skips.append(skip)
    
    outputs = []
    def output_block():
        return Sequential([
            layers.Conv3D(filters=1, kernel_size=1, strides=1, padding='same', use_bias=True),
            layers.Activation('sigmoid', dtype='float32')
        ])
    
    for i in range(len(n_filters)-1):
        if use_deepSup[-1-i]: outputs.append(output_block()(x))
        x_up = layers.Conv3DTranspose(filters=n_filters[-2-i], kernel_size=downUp_spacings[-1-i], strides=downUp_spacings[-1-i], padding='same', use_bias=False)(x)
        x_encoded = skips[-1-i]
        x_scaled = Attention_block(downUp_spacings[-1-i])([x_encoded,x])
        x = layers.Concatenate()([x_up,x_scaled])
        x = Conv_block(n_filters=n_filters[-2-i], kernel_size=kernel_sizes[-2-i], strides1=1, dropout=False)(x)
        
    outputs.append(output_block()(x))
    model = Model(inputs=inputs, outputs=outputs[::-1])
    model.config = {
        'input_shape':input_shape,
        'unet_type':unet_type,
        'deep_supervision':deep_supervision,
        'dropout_deepLayers':dropout_deepLayers,
    }
    return model