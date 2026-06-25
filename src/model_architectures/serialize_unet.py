import json
from tensorflow.keras.models import Model

from .unet import UNet
from ..util import InputShape, UNetType

def save_model_config(
    model: Model,
    config_file: str
) -> None:
    d = {
        'input_shape':model.input_shape.value,
        'unet_type':model.unet_type.value,
        'deep_supervision':model.deep_supervision,
        'dropout_deepLayers':model.dropout_deepLayers
    }
    with open(config_file, 'w') as f: json.dump(d, f, indent=4)
    
def model_from_config(
    config_file: str
) -> Model:
    with open(config_file, 'r') as f:
        d = json.load(f)
    return UNet(
        input_shape = InputShape(tuple(d['input_shape'])),
        unet_type = UNetType(d['unet_type']),
        deep_supervision = d['deep_supervision'],
        dropout_deepLayers = d['dropout_deepLayers']
    )
        