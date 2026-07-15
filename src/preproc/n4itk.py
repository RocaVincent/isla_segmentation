import nibabel as nib
import numpy as np
import ants


def fix_affine_precision(mri:nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    affine = mri.affine.copy()
    zooms = np.array(mri.header.get_zooms()[:3])
    for i in range(3):
        col = affine[:3, i]
        norm = np.linalg.norm(col)
        if norm > 0:
            affine[:3, i] = col / norm * zooms[i]
    return nib.Nifti1Image(mri.get_fdata(dtype=np.float32), affine, mri.header)


def n4BiasFieldCorrection(
    mri:nib.nifti1.Nifti1Image,
    mask:nib.nifti1.Nifti1Image
) -> np.ndarray:
    mri_ants = ants.from_nibabel_nifti(fix_affine_precision(mri))
    mask_ants = ants.from_nibabel_nifti(fix_affine_precision(mask))
    mri = ants.n4_bias_field_correction(mri_ants, mask=mask_ants, rescale_intensities=True, shrink_factor=4, convergence={"iters": [50, 50, 50, 50], "tol": 0})
    return mri.numpy()
    