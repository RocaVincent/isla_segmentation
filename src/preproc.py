from typing import TypedDict
import numpy as np
from scipy import ndimage
import tempfile
import subprocess
import os

from .model_architectures.unet import InputShape

class SideIntegers3D(TypedDict):
    r:int
    l:int
    p:int
    a:int
    i:int
    s:int
    
def n_zeros_slice(
    mri:np.ndarray
) -> SideIntegers3D:
    d11 = 0
    while d11<mri.shape[0]:
        if np.count_nonzero(mri[d11,:,:]) != 0: break
        d11 += 1
    d12 = 0
    while d12<mri.shape[0]:
        if np.count_nonzero(mri[mri.shape[0]-1-d12,:,:]) != 0: break
        d12 += 1
    d21 = 0
    while d21<mri.shape[1]:
        if np.count_nonzero(mri[:,d21,:]) != 0: break
        d21 += 1
    d22 = 0
    while d22<mri.shape[1]:
        if np.count_nonzero(mri[:,mri.shape[1]-1-d22,:]) != 0: break
        d22 += 1
    d31 = 0
    while d31<mri.shape[2]:
        if np.count_nonzero(mri[:,:,d31]) != 0: break
        d31 += 1
    d32 = 0
    while d32<mri.shape[2]:
        if np.count_nonzero(mri[:,:,mri.shape[2]-1-d32]) != 0: break
        d32 += 1
    return {
        'r':d11, 'l':d12,
        'p':d21, 'a':d22,
        'i':d31, 's':d32,
    }
    
def padcrop_numpy(
    data:np.ndarray,
    padcrops: SideIntegers3D
) -> np.ndarray:
    # cropping
    data = data[-min(0,padcrops['r']):data.shape[0]+min(0,padcrops['l']),
                -min(0,padcrops['p']):data.shape[1]+min(0,padcrops['a']),
                -min(0,padcrops['i']):data.shape[2]+min(0,padcrops['s'])]
    
    # padding
    return np.pad(data, (
        (max(0,padcrops['r']),max(0,padcrops['l'])),
        (max(0,padcrops['p']),max(0,padcrops['a'])),
        (max(0,padcrops['i']),max(0,padcrops['s']))
    ))

def padcrop_values(
    len_src:int,
    len_target:int,
    nzeros1:int, 
    nzeros2:int
) -> tuple[int,int]:
    diff_len = len_target-len_src # <0 cropping, >0 padding
    rmd_zeros = nzeros1+nzeros2+diff_len # nb of zeros after padding/cropping
    rmd_zeros_div = rmd_zeros//2 # nb of remaining zeros on each side
    rmd_zeros_rest = rmd_zeros%2
    crop1 = rmd_zeros_div-nzeros1 # <0 for cropping, >0 for padding
    crop2 = rmd_zeros_div-nzeros2 # <0 for cropping, >0 for padding
    if crop1<crop2:  crop1+=rmd_zeros_rest
    else:            crop2+=rmd_zeros_rest
    return crop1,crop2

def resample_padcrop_normalize(
    dwi:np.ndarray,
    adc:np.ndarray,
    brain_mask:np.ndarray,
    factors: tuple[float,float,float],
    target_shape: InputShape
) -> tuple[np.ndarray, np.ndarray, SideIntegers3D]:
    if factors[-1] >= 1.5:
        dwi = ndimage.zoom(dwi, (factors[0],factors[1],1), order=3, prefilter=True)
        dwi = ndimage.zoom(dwi, (1.0, 1.0, factors[2]), order=0, prefilter=False)
        adc = ndimage.zoom(adc, (factors[0],factors[1],1), order=3, prefilter=True)
        adc = ndimage.zoom(adc, (1.0, 1.0, factors[2]), order=0, prefilter=False)
    else:
        dwi = ndimage.zoom(dwi, factors, order=3, prefilter=True)
        adc = ndimage.zoom(adc, factors, order=3, prefilter=True)
    brain_mask = ndimage.zoom(brain_mask, factors, order=0, prefilter=False)
    
    nzeros:SideIntegers3D = n_zeros_slice(brain_mask)
    padcrop_right, padcrop_left = padcrop_values(brain_mask.shape[0], target_shape.value[0], nzeros1=nzeros['r'], nzeros2=nzeros['l'])
    padcrop_post, padcrop_ant = padcrop_values(brain_mask.shape[1], target_shape.value[1], nzeros1=nzeros['p'], nzeros2=nzeros['a'])
    padcrop_inf, padcrop_sup = padcrop_values(brain_mask.shape[2], target_shape.value[2], nzeros1=nzeros['i'], nzeros2=nzeros['s'])
    padcrops:SideIntegers3D = {
        'r':padcrop_right, 'l':padcrop_left,
        'p':padcrop_post, 'a':padcrop_ant,
        'i':padcrop_inf, 's':padcrop_sup
    }
    
    brain_mask = padcrop_numpy(brain_mask, padcrops)
    mask = brain_mask>0
    
    dwi = padcrop_numpy(dwi, padcrops)
    dwi[mask] = (dwi[mask]-dwi[mask].mean()) / dwi[mask].std()
    dwi[~mask] = 0
    np.clip(dwi, a_min=-5, a_max=5, out=dwi)
    
    adc = padcrop_numpy(adc, padcrops)
    adc[mask] = (adc[mask]-adc[mask].mean()) / adc[mask].std()
    adc[~mask] = 0
    np.clip(adc, a_min=-5, a_max=5, out=adc)
    
    return dwi,adc,padcrops
    
    
def skullstrip_n4(
    dwi_path:str,
    adc_path:str,
    out_dwi:str,
    out_adc:str,
    out_brainMask:str
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dwi_brain = f"{tmp}/dwi_brain.nii.gz"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        subprocess.run(["mri_synthstrip", "-i", dwi_path, "-o", dwi_brain, "-m", out_brainMask], check=True, env=env, capture_output=True)
        subprocess.run(['mri_mask',adc_path,out_brainMask,out_adc], check=True, capture_output=True)
        subprocess.run(['N4BiasFieldCorrection','-d','3','-i',dwi_brain,'-x',out_brainMask,'-o',out_dwi], check=True)