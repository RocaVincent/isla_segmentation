import tensorflow as tf

from ..model_architectures.serialize_unet import model_from_config
from .predict import MultiPredictor_flipping


def segment_and_save(
    dwi_paths: list[str],
    adc_paths: list[str],
    brainMask_paths: list[str],
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
    for i,(dwi_path, adc_path, brainMask_path, output_path) in enumerate(zip(dwi_paths,adc_paths,brainMask_paths,output_paths), start=1):
        print(f"Running inference {i}/{len(dwi_paths)}", end='\r')
        predictor.predict(
            dwi_path=dwi_path,
            adc_path=adc_path,
            brainMask_path=brainMask_path,
            output_path=output_path
        )