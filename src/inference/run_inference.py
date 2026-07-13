import tensorflow as tf
from multiprocessing import Pool
import tempfile
import nibabel as nib
import subprocess

from ..model_architectures.serialize_unet import model_from_config
from ..preproc import skullstrip_n4
from .predict import MultiPredictor_flipping


def segment_and_save_preproc(
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
    predictor = MultiPredictor_flipping(models)
    
    print('Running inference...')
    for i,(dwi_path, adc_path, brainMask_path, output_path) in enumerate(zip(dwi_paths,adc_paths,brainMask_paths,output_paths), start=1):
        print(f"Running inference {i}/{len(dwi_paths)}", end='\r')
        predictor.predict(
            dwi_path=dwi_path,
            adc_path=adc_path,
            brainMask_path=brainMask_path,
            output_path=output_path
        )
    print()
    
    
def segment_and_save_raw(
    dwi_paths: list[str],
    adc_paths: list[str],
    output_paths: list[str],
    config_paths: list[str],
    weights_paths: list[str],
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
    predictor = MultiPredictor_flipping(models, gpu_synthstrip=False)
    
    for i,(dwi_path, adc_path, output_path) in enumerate(zip(dwi_paths,adc_paths,output_paths), start=1):
        print(f"Running inference {i}/{len(dwi_paths)}")
        predictor.predict(
            dwi_path=dwi_path,
            adc_path=adc_path,
            output_path=output_path
        )
    print('\n\n')

# def segment_and_save_raw(
#     dwi_paths: list[str],
#     adc_paths: list[str],
#     output_paths: list[str],
#     config_paths: list[str],
#     weights_paths: list[str],
#     use_gpu: bool,
#     nprocs: int
# ) -> None:
#     with tempfile.TemporaryDirectory() as tmp:
#         print('Running preprocessing (SynthStrip and N4ITK)...')
#         dwi_preproc_paths = [f"{tmp}/dwi_preproc{i}.nii.gz" for i in range(1, len(dwi_paths)+1)]
#         adc_preproc_paths = [f"{tmp}/adc_preproc{i}.nii.gz" for i in range(1, len(adc_paths)+1)]
#         brain_masks = [f"{tmp}/brainMask{i}.nii.gz" for i in range(1, len(adc_paths)+1)]
#         with Pool(nprocs) as pool:
#             pool.starmap(skullstrip_n4, zip(dwi_paths, adc_paths, dwi_preproc_paths, adc_preproc_paths, brain_masks))
#         seg_outputs = [f"{tmp}/seg{i}.nii.gz" for i in range(1, len(dwi_paths)+1)]
#         segment_and_save_preproc(
#             dwi_paths=dwi_preproc_paths,
#             adc_paths=adc_preproc_paths,
#             brainMask_paths=brain_masks,
#             output_paths=seg_outputs,
#             config_paths=config_paths,
#             weights_paths=weights_paths,
#             use_gpu=use_gpu
#         )
        