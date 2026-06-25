import tensorflow as tf

from ..model_architectures.serialize_unet import model_from_config
from ..util import ModelInput, ModelConfig
from .predict import MultiPredictor_flipping


    

def segment_and_save(
    model_inputs: list[ModelInput],
    output_paths: list[str],
    config_paths: list[str],
    weights_paths: list[str],
    use_gpu: bool
) -> None:
    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus)==0: print("WARNING: No GPU detected. CPU inference in full precision will be used.")
        else:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print('GPU inference with mixed precision will be used.')
    else:
        tf.config.set_visible_devices([], "GPU")
        print('GPU deactivate, CPU inference in full precision will be used.')
        
    
    print('Model instantiation and weight loading...')
    models = [model_from_config(cfg) for cfg in config_paths]
    for m,w in zip(models,weights_paths): m.load_weights(w)
    predictor = MultiPredictor_flipping([m for m in models])
    
    print('Running inference...')
    for i,(model_input,output_path) in enumerate(zip(model_inputs,output_paths), start=1):
        print(f"Running inference {i}/{len(model_inputs)}", end='\r')
        predictor.predict(**model_input, output_path=output_path)