import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy import ndimage
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import time

from ..model_architectures.unet import InputShape, InputResol
from ..preproc.synthstrip import CustomSynthstrip
from ..preproc.resample_padcrop_normalize import padcrop_numpy, resample_padcrop_normalize
from ..preproc.n4itk import n4BiasFieldCorrection

TARGET_ORNT = nib.orientations.axcodes2ornt(('L', 'A', 'S'))


class SinglePredictor_flipping:
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.input_shape = model.config['input_shape']
        self.deep_sup = model.config['deep_supervision']
        # one volume in, flip-TTA built inside the graph
        self.predict = tf.function(
            self._predict,
            input_signature=(tf.TensorSpec([1, None, None, None, 2], tf.float32),),
            jit_compile=True,
        )

    def _predict(self, mri):
        x = tf.concat([mri, tf.reverse(mri, axis=[1])], axis=0)   # axis 1 = X (axis 0 is batch)
        preds = self.model(x, training=False)
        if self.deep_sup:
            preds = preds[0]
        pred = (preds[0] + tf.reverse(preds[1], axis=[0])) / 2     # un-flip X
        return pred[..., 0]


class MultiPredictor_flipping:
    
    def __init__(self, models: Sequence[tf.keras.Model], n_prefetch):
        self.predictors = [SinglePredictor_flipping(m) for m in models]
        self.groups = {}   # shape key -> (resol, [predictors])
        for p in self.predictors:
            self.groups.setdefault(p.input_shape, []).append(p)
        self.synthstrip = CustomSynthstrip()
        self.n_prefetch = n_prefetch

    # ---------- stage 1: CPU preprocessing (runs in a worker thread) ----------
    def preprocess(self, dwi_path, adc_path):
        ref = nib.load(dwi_path)
        dwi, brain_mask = self.synthstrip.skullstrip(ref)
        adc = nib.load(adc_path).get_fdata(dtype=np.float32) * brain_mask

        dwi = nib.Nifti1Image(dwi, affine=ref.affine, header=ref.header)
        mask_nii = nib.Nifti1Image(brain_mask, affine=ref.affine, header=ref.header)
        dwi = n4BiasFieldCorrection(dwi, mask_nii)

        orig_ornt = nib.orientations.io_orientation(ref.affine)
        input_resol = np.array(ref.header.get_zooms())
        to_reorient = (orig_ornt != TARGET_ORNT).any()
        bwd = None
        if to_reorient:
            fwd = nib.orientations.ornt_transform(orig_ornt, TARGET_ORNT)
            bwd = nib.orientations.ornt_transform(TARGET_ORNT, orig_ornt)
            dwi = nib.orientations.apply_orientation(dwi, fwd)
            adc = nib.orientations.apply_orientation(adc, fwd)
            brain_mask = nib.orientations.apply_orientation(brain_mask, fwd)
            input_resol = input_resol[fwd[:, 0].astype(int)]

        inputs = {}
        for shape_key, resol_key in ((InputShape.SHAPE1, InputResol.RESOL1),
                                     (InputShape.SHAPE2, InputResol.RESOL2)):
            if shape_key not in self.groups:
                continue
            factors = input_resol / resol_key.value
            d, a, padcrops = resample_padcrop_normalize(
                dwi, adc, brain_mask, factors=tuple(factors), target_shape=shape_key)
            mri = np.stack((d, a), axis=3)[None].astype(np.float32)        # [1,X,Y,Z,2]
            inv_padcrops = {k: -v for k, v in padcrops.items()}
            inv_zoom = dwi.shape / np.round(factors * dwi.shape)
            inputs[shape_key] = (mri, inv_padcrops, inv_zoom)

        return dict(ref=ref, out_shape=dwi.shape, inputs=inputs, bwd=bwd)

    # ---------- stage 2: GPU inference + inverse resampling ----------
    def infer(self, ctx):
        n_models = len(self.predictors)
        total = np.zeros(ctx['out_shape'], dtype=np.float32)
        for shape_key, group in self.groups.items():
            mri, inv_padcrops, inv_zoom = ctx['inputs'][shape_key]
            x = tf.constant(mri)
            # dispatch all models asynchronously, average on GPU, single host sync
            pred = (tf.add_n([p.predict(x) for p in group]) / len(group)).numpy()
            pred = padcrop_numpy(pred, inv_padcrops)
            pred = ndimage.zoom(pred.astype(np.float32), inv_zoom, order=1, prefilter=False)
            total += pred * (len(group) / n_models)
        return np.round(total).astype(np.uint8)

    # ---------- stage 3: postprocess + save (worker thread) ----------
    def save(self, ctx, final_pred, output_path):
        if ctx['bwd'] is not None:
            final_pred = nib.orientations.apply_orientation(final_pred, ctx['bwd'])
        hdr = ctx['ref'].header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(final_pred), ctx['ref'].affine, hdr),
                 output_path)

    # ---------- single case ----------
    def predict(self, dwi_path, adc_path, output_path):
        ctx = self.preprocess(dwi_path, adc_path)
        self.save(ctx, self.infer(ctx), output_path)

    # ---------- whole dataset, pipelined ----------
    def predict_all(self, cases):
        """cases: list of (dwi_path, adc_path, output_path)"""
        n = len(cases)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(self.n_prefetch) as pre, ThreadPoolExecutor(1) as post:
            pre_futs = [pre.submit(self.preprocess, d, a) for d, a, _ in cases[:self.n_prefetch]]
            save_futs = []
            for i, (_, _, out) in enumerate(cases):
                ctx = pre_futs[i].result()
                pre_futs[i] = None
                if i + self.n_prefetch < n:
                    d, a, _ = cases[i + self.n_prefetch]
                    pre_futs.append(pre.submit(self.preprocess, d, a))
                pred = self.infer(ctx)
                save_futs.append(post.submit(self.save, ctx, pred, out))
    
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f'[{i + 1}/{n}] inferred | elapsed {elapsed / 60:.1f} min | '
                      f'ETA {eta / 60:.1f} min', flush=True)
    
            for j, f in enumerate(save_futs):
                f.result()
            print(f'Done: {n} cases saved in {(time.perf_counter() - t0) / 60:.1f} min', flush=True)