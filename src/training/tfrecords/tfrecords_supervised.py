from collections.abc import Sequence
import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy import ndimage

from ...preproc.synthstrip import CustomSynthstrip
from ...model_architectures.unet import InputShape, InputResol
from ...preproc.resample_padcrop_normalize import resample_padcrop_normalize, padcrop_numpy
from ...preproc.n4itk import n4BiasFieldCorrection

def create_tfrecords(
    dwi_paths: Sequence[str],
    adc_paths: Sequence[str],
    lesion_paths: Sequence[str],
    target_shape: InputShape,
    dest_dir: str,
    nfiles: int,
    compression: bool = True
) -> None:
    synthstrip = CustomSynthstrip()
    TARGET_ORNT = nib.orientations.axcodes2ornt(('L','A','S'))
    if target_shape==InputShape.SHAPE1: target_resol = np.array(InputResol.RESOL1)
    else:                               target_resol = np.array(InputResol.RESOL2)
        
    def write_tfrecordsfiles(dwi_paths, adc_paths, lesion_paths, tfrecord_dest):
        with tf.io.TFRecordWriter(tfrecord_dest, options='GZIP' if compression else None) as record_f:
            for dwi_path, adc_path, lesion_path in zip(dwi_paths, adc_paths, lesion_paths):
                # skullstrip
                ref = nib.load(dwi_path)
                dwi,brain_mask = synthstrip.skullstrip(ref)
                adc = nib.load(adc_path).get_fdata()
                adc = adc*brain_mask
                lesion = nib.load(lesion_path).get_fdata()
    
                #N4ITK
                dwi = nib.Nifti1Image(dwi, affine=ref.affine, header=ref.header)
                brain_mask_nifti = nib.Nifti1Image(brain_mask, affine=ref.affine, header=ref.header)
                dwi = n4BiasFieldCorrection(dwi, brain_mask_nifti)
    
                # ORIENTATION
                orig_ornt = nib.orientations.io_orientation(ref.affine)
                input_resol = np.array(ref.header.get_zooms())
                if (orig_ornt!=TARGET_ORNT).any():
                    fwd_reorient = nib.orientations.ornt_transform(orig_ornt,TARGET_ORNT)
                    dwi = nib.orientations.apply_orientation(dwi,fwd_reorient)
                    adc = nib.orientations.apply_orientation(adc,fwd_reorient)
                    brain_mask = nib.orientations.apply_orientation(brain_mask,fwd_reorient)
                    lesion = nib.orientations.apply_orientation(lesion,fwd_reorient)
                    axis_order = fwd_reorient[:, 0].astype(int)
                    input_resol = input_resol[axis_order]
                    
    
                # RESAMPLE,PADCROP,NORMALIZE
                factors = input_resol / target_resol
                dwi,adc,padcrops = resample_padcrop_normalize(
                    dwi=dwi, adc=adc, brain_mask=brain_mask, factors=factors, target_shape=target_shape
                )
                lesion = ndimage.zoom(lesion, factors, order=0, prefilter=False)
                lesion = padcrop_numpy(lesion, padcrops)
    
                # Ecriture record
                mris = np.stack((dwi,adc), axis=3)
                features = {
                    'mris': tf.train.Feature(float_list=tf.train.FloatList(value=mris.flatten())),
                    'lesion': tf.train.Feature(float_list=tf.train.FloatList(value=lesion.flatten())),
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                record_f.write(example.SerializeToString())

        
    n,mod = divmod(len(dwi_paths),nfiles)
    for i in range(mod):
        print(f"Writing record files {i+1}/{nfiles}", end='\r')
        records_path = f"{dest_dir}/records{i+1}.tfrecord"
        if compression: records_path += '.gz'
        idx_start = i*(n+1)
        idx_end = (i+1)*(n+1)
        write_tfrecordsfiles(
            dwi_paths=dwi_paths[idx_start:idx_end],
            adc_paths=adc_paths[idx_start:idx_end],
            lesion_paths=lesion_paths[idx_start:idx_end],
            tfrecord_dest=records_path
        )

    for i in range(nfiles-mod):
        print(f"Writing record files {mod+i+1}/{nfiles}", end='\r')
        records_path = f"{dest_dir}/records{mod+i+1}.tfrecord"
        if compression: records_path += '.gz'
        idx_start = mod*(n+1)+i*n
        idx_end = mod*(n+1)+(i+1)*n
        write_tfrecordsfiles(
            dwi_paths=dwi_paths[idx_start:idx_end],
            adc_paths=adc_paths[idx_start:idx_end],
            lesion_paths=lesion_paths[idx_start:idx_end],
            tfrecord_dest=records_path
        )