import tensorflow as tf
import nibabel as nib
from scipy import ndimage
import numpy as np
import ants
from collections.abc import Sequence

from ..model_architectures.unet import InputShape, InputResol
from ..preproc.synthstrip import CustomSynthstrip
from ..preproc.resample_padcrop_normalize import padcrop_numpy, resample_padcrop_normalize
from ..preproc.n4itk import n4BiasFieldCorrection


class SinglePredictor_flipping:
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.input_shape = model.config['input_shape']
        
    @tf.function(input_signature=(
      tf.TensorSpec(shape=[2,None,None,None,2], dtype='float32'),  
    ), jit_compile=True)
    def predict(self, mris:tf.Tensor) -> tf.Tensor:
        preds = self.model(mris, training=False)
        if self.model.config['deep_supervision']: preds = preds[0]
        pred2 = tf.reverse(preds[1], axis=[0])
        pred = (preds[0]+pred2)/2
        return pred[:,:,:,0]
    
TARGET_ORNT = nib.orientations.axcodes2ornt(('L','A','S'))
class MultiPredictor_flipping:
    
    def __init__(
        self,
        models: Sequence[tf.keras.Model]
    ):
        self.predictors = [
            SinglePredictor_flipping(model=model) for model in models
        ]
        self.shape1 = np.any([predictor.input_shape==InputShape.SHAPE1 for predictor in self.predictors])
        self.shape2 = np.any([predictor.input_shape==InputShape.SHAPE2 for predictor in self.predictors])
        self.synthstrip = CustomSynthstrip()
        
    def predict(self, dwi_path:str, adc_path:str, output_path:str) -> None:
        ref = nib.load(dwi_path)
        
        # synthstrip
        print('Running Synthstrip...')
        dwi,brain_mask = self.synthstrip.skullstrip(ref)
        adc = nib.load(adc_path).get_fdata()
        adc = adc*brain_mask.get_fdata()
        
        # N4ITK
        print('Running N4ITK...')
        dwi = n4BiasFieldCorrection(dwi,brain_mask)
        brain_mask = brain_mask.get_fdata()
        
        print('Reorientation and resampling...')
        # ORIENTATION
        orig_ornt = nib.orientations.io_orientation(ref.affine)
        to_reorient = (orig_ornt!=TARGET_ORNT).any()
        if to_reorient:
            fwd_reorient = nib.orientations.ornt_transform(orig_ornt,TARGET_ORNT)
            bwd_reorient = nib.orientations.ornt_transform(TARGET_ORNT,orig_ornt)
            dwi = nib.orientations.apply_orientation(dwi,fwd_reorient)
            adc = nib.orientations.apply_orientation(adc,fwd_reorient)
            brain_mask = nib.orientations.apply_orientation(brain_mask,fwd_reorient)
        
        # RESAMPLING, PADCROP, NORMALIZATION
        input_resol = np.array(ref.header.get_zooms())
        if self.shape1:
            factors = input_resol / InputResol.RESOL1.value
            dwi1,adc1,padcrops1 = resample_padcrop_normalize(dwi,adc,brain_mask,
                                                             factors=tuple(factors), target_shape=InputShape.SHAPE1)
            mris1 = np.stack((dwi1,adc1), axis=3)
            mris1_flip = np.flip(mris1, axis=0)
            mris1 = np.stack((mris1,mris1_flip))
            mris1 = tf.constant(mris1, dtype='float32')
            inverse_padcrops1 = {k:-v for k,v in padcrops1.items()}
            resample_shape = np.round(factors*dwi.shape)
            inverse_zoom1 = dwi.shape / resample_shape
        if self.shape2:
            factors = input_resol / InputResol.RESOL2.value
            dwi2,adc2,padcrops2 = resample_padcrop_normalize(dwi,adc,brain_mask,
                                                             factors=tuple(factors), target_shape=InputShape.SHAPE2)
            mris2 = np.stack((dwi2,adc2), axis=3)
            mris2_flip = np.flip(mris2, axis=0)
            mris2 = np.stack((mris2,mris2_flip))
            mris2 = tf.constant(mris2, dtype='float32')
            inverse_padcrops2 = {k:-v for k,v in padcrops2.items()}
            resample_shape = np.round(factors*dwi.shape)
            inverse_zoom2 = dwi.shape / resample_shape
         
        # PREDICTIONS
        print('Inferences...')
        sum_preds = np.zeros(dwi.shape)
        for predictor in self.predictors:
            if predictor.input_shape==InputShape.SHAPE1:
                inp = mris1
                inverse_padcrops = inverse_padcrops1
                inverse_zoom = inverse_zoom1
            else:
                inp = mris2
                inverse_padcrops = inverse_padcrops2
                inverse_zoom = inverse_zoom2
            pred = predictor.predict(inp).numpy()
            pred = padcrop_numpy(pred, inverse_padcrops)
            pred = ndimage.zoom(pred, inverse_zoom, order=1, prefilter=False)
            sum_preds += pred
        final_pred = np.round(sum_preds / len(self.predictors)).astype('int')
        if to_reorient:
            final_pred = nib.orientations.apply_orientation(final_pred, bwd_reorient)
        final_pred = nib.Nifti1Image(final_pred, affine=ref.affine, header=ref.header)
        nib.save(final_pred, output_path)