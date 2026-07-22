import tensorflow as tf
from collections.abc import Callable, Sequence

from .util import downsample_masks

import json,os
IMAGE_SHAPE = json.loads(os.environ['IMAGE_SHAPE'])


class FullySupervisedTrainer:

    def __init__(
        self,
        model: tf.keras.Model,
        loss_fn: Callable[[tf.Tensor,tf.Tensor],tf.Tensor],
        ds_weights: Sequence[float],
        ls_optimizer: tf.keras.mixed_precision.LossScaleOptimizer,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.ds_weights = tf.constant(ds_weights, dtype='float32')
        self.ls_optimizer = ls_optimizer

        # computing output shapes for dataset indices
        output_shapes = model.compute_output_shape([None]+IMAGE_SHAPE+[2])
        assert len(output_shapes)==len(ds_weights)
        assert output_shapes[0] == [None]+IMAGE_SHAPE+[1]
        ds_shapes = []
        for shape in output_shapes[1:]:
            assert shape[0] is None and shape[4]==1
            ds_shapes.append(shape[1:4])

        # downsampling indices for each deep supervision resolution
        self.w_indices = [tf.cast(tf.round(tf.linspace(0,IMAGE_SHAPE[0]-1, shape[0])), dtype='int32') for shape in ds_shapes]
        self.h_indices = [tf.cast(tf.round(tf.linspace(0,IMAGE_SHAPE[1]-1, shape[1])), dtype='int32') for shape in ds_shapes]
        self.d_indices = [tf.cast(tf.round(tf.linspace(0,IMAGE_SHAPE[2]-1, shape[2])), dtype='int32') for shape in ds_shapes]

        self.n_ds = len(ds_shapes)
        

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None]+IMAGE_SHAPE+[2], dtype='float32'),
        tf.TensorSpec(shape=[None]+IMAGE_SHAPE+[1], dtype='float32')
    ), jit_compile=True)
    def train_step(self, mris, masks):
        down_masks = [downsample_masks(masks, self.w_indices[i], self.h_indices[i], self.d_indices[i]) for i in range(self.n_ds)]
        with tf.GradientTape() as tape:
            preds = self.model(mris, training=True)
            loss1 = self.loss_fn(masks,preds[0])
            losses_aux = [self.loss_fn(down_masks[i], preds[i+1]) for i in range(self.n_ds)]
            loss = loss1*self.ds_weights[0] + tf.math.reduce_sum(tf.stack(losses_aux)*self.ds_weights[1:])
            loss_ = self.ls_optimizer.get_scaled_loss(loss)
        grads = tape.gradient(loss_, self.model.trainable_variables)
        grads = self.ls_optimizer.get_unscaled_gradients(grads)
        self.ls_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss