import tensorflow as tf
import nibabel as nib
from scipy import ndimage
import numpy as np

from ..model_architectures.unet import InputShape, InputResol
from ..preproc import padcrop_numpy, resample_padcrop_normalize


class SinglePredictor_flipping:
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.input_shape = model.config['input_shape']
        
    @tf.function(input_signature=(
      tf.TensorSpec(shape=[2,None,None,None,2], dtype='float32'),  
    ), jit_compile=True)
    def predict(self, mris:tf.Tensor):
        preds = self.model(mris, training=False)
        if self.model.config['deep_supervision']: preds = preds[0]
        pred2 = tf.reverse(preds[1], axis=[0])
        pred = (preds[0]+pred2)/2
        return pred[:,:,:,0]
    
    
class MultiPredictor_flipping:
    
    def __init__(
        self,
        models: list[tf.keras.Model]
    ):
        self.predictors = [
            SinglePredictor_flipping(model=model) for model in models
        ]
        self.shape1 = np.any([predictor.input_shape==InputShape.SHAPE1 for predictor in self.predictors])
        self.shape2 = np.any([predictor.input_shape==InputShape.SHAPE2 for predictor in self.predictors])
        
    def predict(self, dwi_path:str, adc_path:str, brainMask_path:str, output_path:str):
        ref = nib.load(dwi_path)
        dwi = ref.get_fdata()
        adc = nib.load(adc_path).get_fdata()
        brain_mask = nib.load(brainMask_path).get_fdata()
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
        final_pred = nib.Nifti1Image(final_pred, affine=ref.affine, header=ref.header)
        nib.save(final_pred, output_path)

    



# class MultiPredictor_flipping:
    
#     def __init__(
#         self,
#         models: list[tf.keras.Model]
#     ):
#         self.predictors = [
#             SinglePredictor_flipping(model=model) for model in models
#         ]
#         self.shape1 = np.any([predictor.input_shape==InputShape.SHAPE1 for predictor in self.predictors])
#         if self.shape1: self.shape1_numpy = np.array(InputShape.SHAPE1.value)
#         self.shape2 = np.any([predictor.input_shape==InputShape.SHAPE2 for predictor in self.predictors])
#         if self.shape2: self.shape2_numpy = np.array(InputShape.SHAPE2.value)
        
#     def predict(self, dwi_path_shape1:str, adc_path_shape1:str, dwi_path_shape2:str, adc_path_shape2:str,
#                       reference_path:str, output_path:str,
#                       padcrops1: PadCropParamsType, padcrops2: PadCropParamsType
#     ):
#         ref = nib.load(reference_path)
#         output_shape = np.array(ref.shape)
#         if self.shape1:
#             with gzip.open(dwi_path_shape1, 'rb') as f: dwi_data1 = np.load(f)
#             with gzip.open(adc_path_shape1, 'rb') as f: adc_data1 = np.load(f)
#             mris1 = np.stack((dwi_data1,adc_data1), axis=3)
#             mris1_flip = np.flip(mris1, axis=0)
#             mris1 = np.stack((mris1,mris1_flip))
#             mris1 = tf.constant(mris1, dtype='float32')
#             shape1_padcrop = self.shape1_numpy + [padcrops1['r']+padcrops1['l'] , padcrops1['p']+padcrops1['a'] , padcrops1['i']+padcrops1['s']]
#             zoom_factor1 = output_shape / shape1_padcrop
#         if self.shape2:
#             with gzip.open(dwi_path_shape2, 'rb') as f: dwi_data2 = np.load(f)
#             with gzip.open(adc_path_shape2, 'rb') as f: adc_data2 = np.load(f)
#             mris2 = np.stack((dwi_data2,adc_data2), axis=3)
#             mris2_flip = np.flip(mris2, axis=0)
#             mris2 = np.stack((mris2,mris2_flip))
#             mris2 = tf.constant(mris2, dtype='float32')
#             shape2_padcrop = self.shape2_numpy + [padcrops2['r']+padcrops2['l'] , padcrops2['p']+padcrops2['a'] , padcrops2['i']+padcrops2['s']]
#             zoom_factor2 = output_shape / shape2_padcrop
        
#         sum_preds = np.zeros(output_shape)
#         for predictor in self.predictors:
#             if predictor.input_shape==InputShape.SHAPE1:
#                 inp = mris1
#                 padcrops = padcrops1
#                 zoom_factor = zoom_factor1
#             else:
#                 inp = mris2
#                 padcrops = padcrops2
#                 zoom_factor = zoom_factor2
#             pred = predictor.predict(inp).numpy()
#             pred = pad_crop_numpy(pred, **padcrops)
#             pred = ndimage.zoom(pred, zoom_factor, order=1, prefilter=False)
#             sum_preds += pred
#         final_pred = np.round(sum_preds / len(self.predictors)).astype('int')
#         final_pred = nib.Nifti1Image(final_pred, affine=ref.affine, header=ref.header)
#         nib.save(final_pred, output_path)
    