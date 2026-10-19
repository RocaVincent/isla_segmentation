import tensorflow as tf
from collections.abc import Sequence

from ..model_architectures.serialize_unet import model_from_config
from .predict import MultiPredictor_flipping
    
    
def segment_and_save_raw(
    dwi_paths: Sequence[str],
    adc_paths: Sequence[str],
    output_paths: Sequence[str],
    config_paths: Sequence[str],
    weights_paths: Sequence[str],
    use_gpu: bool
):
    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus)==0: print("WARNING: No GPU detected. CPU inference in full precision will be used.")
        else:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print('GPU inference with mixed precision will be used.')
    else:
        tf.config.set_visible_devices([], "GPU")
        print('GPU deactivate, CPU inference in full precision will be used.')
        
    print('Model instantiation and weight loading...')
    models = [model_from_config(cfg) for cfg in config_paths]
    for m,w in zip(models,weights_paths): m.load_weights(w)
    predictor = MultiPredictor_flipping(models)
    predictor.predict_all(list(zip(dwi_paths,adc_paths,output_paths)))
        