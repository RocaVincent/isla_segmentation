from scipy import ndimage
from skimage import transform
from elasticdeform import deform_grid
import tensorflow as tf

import json,os
IMAGE_SHAPE = json.loads(os.environ['IMAGE_SHAPE'])


def rotate_numpy(mris, angle, order):
    return ndimage.rotate(mris, angle=angle, axes=(0,1), reshape=False, order=order, mode='constant', cval=0)

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[None], dtype='float32'),
    tf.TensorSpec(shape=(), dtype='float32'),
    tf.TensorSpec(shape=(), dtype='int32')
))
def rotate_tensor(mris, angle, order):
    return tf.numpy_function(rotate_numpy, inp=[mris, angle, order], Tout='float32', stateful=False)



def scale_numpy(mris, factor, order):
    return ndimage.zoom(mris, (factor, factor, 1, 1), order=order, mode='constant', cval=0, prefilter=False)

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[None], dtype='float32'),
    tf.TensorSpec(shape=(), dtype='float32'),
    tf.TensorSpec(shape=(), dtype='int32')
))
def scale_tensor(mris, factor, order):
    mris = tf.numpy_function(scale_numpy, inp=[mris, factor, order], Tout='float32', stateful=False)
    new_shape = tf.shape(mris)[:-2] # z and channel axis unchanged
    if factor>1:
        ns_crop = new_shape-IMAGE_SHAPE[:-1]
        c0_1 = ns_crop[0]//2
        c0_2 = c0_1+(ns_crop[0]%2)
        c1_1 = ns_crop[1]//2
        c1_2 = c1_1+(ns_crop[1]%2)
        mris = mris[c0_1:new_shape[0]-c0_2,c1_1:new_shape[1]-c1_2,:,:]
    else:
        ns_pad = IMAGE_SHAPE[:-1]-new_shape
        c0_1 = ns_pad[0]//2
        c0_2 = c0_1+(ns_pad[0]%2)
        c1_1 = ns_pad[1]//2
        c1_2 = c1_1+(ns_pad[1]%2)
        mris = tf.pad(mris, paddings=(
          (c0_1,c0_2), (c1_1,c1_2), (0,0), (0,0)
        ), mode='CONSTANT', constant_values=0)
    return mris


@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[N_CHANNELS], dtype='float32'),
    tf.TensorSpec(shape=(IMAGE_SHAPE+[1]), dtype='float32'),
))
def add_noise(mris, noise):
    return tf.where(tf.math.abs(mris)>1e-4, mris+noise, mris)



def elastic_deform_mrisMask_numpy(mris, mask, displacement):
    return deform_grid([mris,mask], displacement=displacement, order=[3,0], axis=(0,1))

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[2], dtype='float32'),
    tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
    tf.TensorSpec(shape=(2,None,None), dtype='float32')
))
def elastic_deform_mrisMask_tensor(mris, mask, displacement):
    return tf.numpy_function(elastic_deform_mrisMask_numpy, inp=[mris,mask,displacement], Tout=['float32','float32'], stateful=False)

def elastic_deform_mris_numpy(mris, displacement):
    return deform_grid(mris, displacement=displacement, order=3, axis=(0,1))

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[2], dtype='float32'),
    tf.TensorSpec(shape=(2,None,None), dtype='float32')
))
def elastic_deform_mris_tensor(mris, displacement):
    return tf.numpy_function(elastic_deform_mris_numpy, inp=[mris,displacement], Tout='float32', stateful=False)



idx1 = tf.range(IMAGE_SHAPE[0], dtype='int32')
idx2 = tf.range(IMAGE_SHAPE[1], dtype='int32')
idx3 = tf.range(IMAGE_SHAPE[2], dtype='int32')
D1 = tf.constant(IMAGE_SHAPE[0], dtype='int32')
D2 = tf.constant(IMAGE_SHAPE[1], dtype='int32')
D3 = tf.constant(IMAGE_SHAPE[2], dtype='int32')
@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[None], dtype='float32'),
    tf.TensorSpec(shape=(3,), dtype='int32')
))
def shift_mris(mris, shifts):
    mris = tf.roll(mris, shift=shifts, axis=[0,1,2])
    mask = (
        ((idx1>=shifts[0])&(idx1<D1+shifts[0]))[:,None,None,None] &
        ((idx2>=shifts[1])&(idx2<D2+shifts[1]))[None,:,None,None] &
        ((idx3>=shifts[2])&(idx3<D3+shifts[2]))[None,None,:,None]
    )
    return tf.where(mask, mris, 0)



@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[2], dtype='float32'),
    tf.TensorSpec(shape=(IMAGE_SHAPE+[1]), dtype='float32'),
))
def add_noise(mris, noise):
    return tf.where(tf.math.abs(mris)>1e-4, mris+noise, mris)



def gauss_filter_numpy(mri, sigma):
    return ndimage.gaussian_filter(mri, (sigma, sigma, 0), order=0, mode='constant', cval=0)

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE, dtype='float32'),
    tf.TensorSpec(shape=(), dtype='float32')
))
def gauss_filter_tensor(mri, sigma):
    return tf.numpy_function(gauss_filter_numpy, inp=[mri, sigma], Tout='float32', stateful=False)



@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[2], dtype='float32'),
    tf.TensorSpec(shape=(), dtype='float32')
))
def update_brightness(mris, factor):
    return mris*factor

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[2], dtype='float32'),
    tf.TensorSpec(shape=(), dtype='float32')
))
def update_contrast(mris, factor):
    v_min = tf.math.reduce_min(mris, axis=(0,1,2), keepdims=True)
    v_max = tf.math.reduce_max(mris, axis=(0,1,2), keepdims=True)
    return tf.clip_by_value(mris*factor, clip_value_min=v_min, clip_value_max=v_max)



def simulate_lowRes_numpy(mri, downsampling_factor):
    tmp_shape = round(IMAGE_SHAPE[0]/downsampling_factor), round(IMAGE_SHAPE[1]/downsampling_factor), IMAGE_SHAPE[2]
    downsample = transform.resize(mri, tmp_shape, order=0, mode='constant', cval=0, clip=False, anti_aliasing=False)
    return transform.resize(downsample, mri.shape, order=3, mode='constant', cval=0, clip=False, anti_aliasing=False)

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE, dtype='float32'),
    tf.TensorSpec(shape=(), dtype='float32')
))
def simulate_lowRes_tensor(mri, downsampling_factor):
    mask = tf.math.abs(mri)>1e-4
    mri2 = tf.numpy_function(simulate_lowRes_numpy, inp=[mri, downsampling_factor], Tout='float32', stateful=False)
    return tf.where(mask, mri2, mri)



@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[2], dtype='float32'),
    tf.TensorSpec(shape=(), dtype='float32'),
    tf.TensorSpec(shape=(), dtype='bool'),
))
def gamma_augmentation(mris, gamma, revert):
    i_min = tf.math.reduce_min(mris, axis=(0,1,2), keepdims=True)
    i_range = tf.math.reduce_max(mris, axis=(0,1,2), keepdims=True)-i_min
    mris = (mris-i_min)/i_range
    if revert: mris = 1 - tf.math.pow(1-mris,gamma)
    else:      mris = tf.math.pow(mris,gamma)
    return mris*i_range + i_min



def augmentation_fns(flip: bool, rotation: bool, scaling: bool, elastic: bool, shift: bool, gaussNoise: bool, gaussBlur: bool,
                     brightness: bool, contrast:bool, lowRes: bool, gamma: bool):
    if rotation:
        angle_max = 30 #20
        proba_rotation = 0.2
    if scaling:
        scaling_min, scaling_max = 0.7,1.4 #0.8,1.2
        proba_scaling = 0.2
    if elastic:
        if IMAGE_SHAPE==[160,192,80]:
            n_points_x = 3
            n_points_y = 3
            sigma_displacement = 4
        elif IMAGE_SHAPE==[256,256,48]:
            n_points_x = 4
            n_points_y = 4
            sigma_displacement = 5
        else: raise Exception(f"Unknow image shape {IMAGE_SHAPE}")
        proba_elastic = 0.2
    if shift:
        shifts_max = 20,20,10
        proba_shift = 0.5#0.3
    if gaussNoise:
        std_gaussNoise_max = 0.3#0.15
        proba_gaussNoise = 0.15
    if gaussBlur:
        sigma_gaussBlur_min, sigma_gaussBlur_max = 0.5,1.5#0.5,1.2
        proba_gaussBlur = 0.2
        proba_gaussBlur_modality = 0.5
    if brightness:
        factor_brightness_min, factor_brightness_max = 0.7, 1.3
        proba_brightness = 0.15
    if contrast:
        factor_contrast_min, factor_contrast_max = 0.65,1.5
        proba_contrast = 0.15
    if lowRes:
        factor_lowRes_min, factor_lowRes_max = 1,2
        proba_lowRes = 0.25
        proba_lowRes_modality = 0.5
    if gamma:
        gamma_min, gamma_max = 0.7,1.5#0.8,1.2
        proba_gamma = 0.15
        proba_gamma_revert = 0.15
    
    
    @tf.function(input_signature=(
        tf.TensorSpec(shape=IMAGE_SHAPE+[None], dtype='float32'),
        tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
    ), jit_compile=not (rotation or scaling or lowRes or elastic))
    def supervised_augmentation(mris, mask):
        if flip and tf.random.uniform(shape=(), minval=0, maxval=1, dtype="float32") < 0.5:
            mris = tf.reverse(mris,[0])
            mask = tf.reverse(mask,[0])
        if rotation and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_rotation:
            angle = tf.random.uniform(shape=(), minval=-angle_max, maxval=angle_max, dtype='float32')
            mris = rotate_tensor(mris, angle, 3)
            mask = rotate_tensor(mask, angle, 0)
        if scaling and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_scaling:
            scaling_factor = tf.random.uniform(shape=(), minval=scaling_min, maxval=scaling_max, dtype='float32')
            mris = scale_tensor(mris, scaling_factor, 1)
            mask = scale_tensor(mask, scaling_factor, 0)
        if elastic and  tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_elastic:
            displacements = tf.random.normal(shape=(2,n_points_x,n_points_y), mean=0, stddev=sigma_displacement, dtype='float32')
            mris,mask = elastic_deform_mrisMask_tensor(mris, mask, displacements)
        if shift and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_shift:
            shifts = (
                tf.random.uniform((), minval=-shifts_max[0], maxval=shifts_max[0]+1, dtype='int32'),
                tf.random.uniform((), minval=-shifts_max[1], maxval=shifts_max[1]+1, dtype='int32'),
                tf.random.uniform((), minval=-shifts_max[2], maxval=shifts_max[2]+1, dtype='int32'),
            )
            mris = shift_mris(mris, shifts)
            mask = shift_mris(mask, shifts)
        if gaussNoise and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussNoise:
            std_gaussNoise = tf.random.uniform(shape=(), minval=0, maxval=std_gaussNoise_max, dtype='float32')
            noise = tf.random.normal(shape=IMAGE_SHAPE+[1], mean=0.0, stddev=std_gaussNoise, dtype='float32')
            mris = add_noise(mris, noise)
        if gaussBlur and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussBlur:
            modalities = []
            for i in range(N_CHANNELS):
                if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussBlur_modality:
                    sigma = tf.random.uniform(shape=(), minval=sigma_gaussBlur_min, maxval=sigma_gaussBlur_max, dtype='float32')
                    modality = gauss_filter_tensor(mris[:,:,:,i], sigma)
                else: modality = mris[:,:,:,i]
                modalities.append(modality)
            mris = tf.stack(modalities, axis=3)
        if brightness and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_brightness:
            factor_brightness = tf.random.uniform(shape=(), minval=factor_brightness_min, maxval=factor_brightness_max, dtype='float32')
            mris = update_brightness(mris, factor_brightness)
        if contrast and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_contrast:
            factor_contrast = tf.random.uniform(shape=(), minval=factor_contrast_min, maxval=factor_contrast_max, dtype='float32')
            mris = update_contrast(mris, factor_contrast)
        if lowRes and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_lowRes:
            modalities = []
            for i in range(N_CHANNELS):
                if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_lowRes_modality:
                    downsampling_factor = tf.random.uniform(shape=(), minval=factor_lowRes_min, maxval=factor_lowRes_max, dtype='float32')
                    modality = simulate_lowRes_tensor(mris[:,:,:,i], downsampling_factor)
                else: modality = mris[:,:,:,i]
                modalities.append(modality)
            mris = tf.stack(modalities, axis=3)
        if gamma and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gamma:
            gamma_coef = tf.random.uniform(shape=(), minval=gamma_min, maxval=gamma_max, dtype='float32')
            gamma_revert = tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gamma_revert
            mris = gamma_augmentation(mris, gamma_coef, gamma_revert)
        return mris,mask
    
    @tf.function(input_signature=(
        tf.TensorSpec(shape=IMAGE_SHAPE+[None], dtype='float32'),
    ), jit_compile=False)
    def unsupervised_augmentation(mris):
        if flip and tf.random.uniform(shape=(), minval=0, maxval=1, dtype="float32") < 0.5:
            mris = tf.reverse(mris,[0])
        if rotation and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_rotation:
            angle = tf.random.uniform(shape=(), minval=-angle_max, maxval=angle_max, dtype='float32')
            mris = rotate_tensor(mris, angle, 3)
        if scaling and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_scaling:
            scaling_factor = tf.random.uniform(shape=(), minval=scaling_min, maxval=scaling_max, dtype='float32')
            mris = scale_tensor(mris, scaling_factor, 1)
        if elastic and  tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_elastic:
            displacements = tf.random.normal(shape=(2,n_points_x,n_points_y), mean=0, stddev=sigma_displacement, dtype='float32')
            mris = elastic_deform_mris_tensor(mris,displacements)
        if shift and tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_shift:
            shifts = (
                tf.random.uniform((), minval=-shifts_max[0], maxval=shifts_max[0]+1, dtype='int32'),
                tf.random.uniform((), minval=-shifts_max[1], maxval=shifts_max[1]+1, dtype='int32'),
                tf.random.uniform((), minval=-shifts_max[2], maxval=shifts_max[2]+1, dtype='int32'),
            )
            mris = shift_mris(mris, shifts)
        
        mris2 = mris
        
        if gaussNoise:
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussNoise:
                std_gaussNoise = tf.random.uniform(shape=(), minval=0, maxval=std_gaussNoise_max, dtype='float32')
                noise = tf.random.normal(shape=IMAGE_SHAPE+[1], mean=0.0, stddev=std_gaussNoise, dtype='float32')
                mris = add_noise(mris, noise)
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussNoise:
                std_gaussNoise = tf.random.uniform(shape=(), minval=0, maxval=std_gaussNoise_max, dtype='float32')
                noise = tf.random.normal(shape=IMAGE_SHAPE+[1], mean=0.0, stddev=std_gaussNoise, dtype='float32')
                mris2 = add_noise(mris2, noise)
        if gaussBlur:
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussBlur:
                modalities = []
                for i in range(2):
                    if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussBlur_modality:
                        sigma = tf.random.uniform(shape=(), minval=sigma_gaussBlur_min, maxval=sigma_gaussBlur_max, dtype='float32')
                        modality = gauss_filter_tensor(mris[:,:,:,i], sigma)
                    else: modality = mris[:,:,:,i]
                    modalities.append(modality)
                mris = tf.stack(modalities, axis=3)
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussBlur:
                modalities = []
                for i in range(2):
                    if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gaussBlur_modality:
                        sigma = tf.random.uniform(shape=(), minval=sigma_gaussBlur_min, maxval=sigma_gaussBlur_max, dtype='float32')
                        modality = gauss_filter_tensor(mris2[:,:,:,i], sigma)
                    else: modality = mris2[:,:,:,i]
                    modalities.append(modality)
                mris2 = tf.stack(modalities, axis=3)
        if brightness:
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_brightness:
                factor_brightness = tf.random.uniform(shape=(), minval=factor_brightness_min, maxval=factor_brightness_max, dtype='float32')
                mris = update_brightness(mris, factor_brightness)
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_brightness:
                factor_brightness = tf.random.uniform(shape=(), minval=factor_brightness_min, maxval=factor_brightness_max, dtype='float32')
                mris2 = update_brightness(mris2, factor_brightness)
        if lowRes:
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_lowRes:
                modalities = []
                for i in range(2):
                    if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_lowRes_modality:
                        downsampling_factor = tf.random.uniform(shape=(), minval=factor_lowRes_min, maxval=factor_lowRes_max, dtype='float32')
                        modality = simulate_lowRes_tensor(mris[:,:,:,i], downsampling_factor)
                    else: modality = mris[:,:,:,i]
                    modalities.append(modality)
                mris = tf.stack(modalities, axis=3)
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_lowRes:
                modalities = []
                for i in range(2):
                    if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_lowRes_modality:
                        downsampling_factor = tf.random.uniform(shape=(), minval=factor_lowRes_min, maxval=factor_lowRes_max, dtype='float32')
                        modality = simulate_lowRes_tensor(mris2[:,:,:,i], downsampling_factor)
                    else: modality = mris2[:,:,:,i]
                    modalities.append(modality)
                mris2 = tf.stack(modalities, axis=3)
        if gamma:
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gamma:
                gamma_coef = tf.random.uniform(shape=(), minval=gamma_min, maxval=gamma_max, dtype='float32')
                gamma_revert = tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gamma_revert
                mris = gamma_augmentation(mris, gamma_coef, gamma_revert)
            if tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gamma:
                gamma_coef = tf.random.uniform(shape=(), minval=gamma_min, maxval=gamma_max, dtype='float32')
                gamma_revert = tf.random.uniform(shape=(), minval=0, maxval=1, dtype='float32') < proba_gamma_revert
                mris2 = gamma_augmentation(mris2, gamma_coef, gamma_revert)
        return mris,mris2
    
    return supervised_augmentation,unsupervised_augmentation