import numpy as np
from typing import TypedDict
from enum import Enum

# Pad crop function
def pad_crop_numpy(data, r, l, p, a, i, s):
    # cropping
    data = data[-min(0,r):data.shape[0]+min(0,l),
                -min(0,p):data.shape[1]+min(0,a),
                -min(0,i):data.shape[2]+min(0,s)]
    
    # padding
    return np.pad(data, (
        (max(0,r),max(0,l)),
        (max(0,p),max(0,a)),
        (max(0,i),max(0,s))
    ))


class PadCropParamsType(TypedDict):
    r:int
    l:int
    p:int
    a:int
    i:int
    s:int
    
class InputShape(Enum):
    SHAPE1 = (160,192,80)
    SHAPE2 = (256,256,48)
    
class UNetType(Enum):
    StdUNet = 'StdUNet'
    ResUNet = 'ResUNet'
    
class ModelConfig(TypedDict):
    input_shape: InputShape
    unet_type: UNetType
    deep_supervision: bool
    dropout_deepLayers: bool
    
class ModelInput(TypedDict):
    dwi_path_shape1: str
    adc_path_shape1: str
    dwi_path_shape2: str
    adc_path_shape2: str
    padcrops1: PadCropParamsType
    padcrops2: PadCropParamsType
    reference_path: str