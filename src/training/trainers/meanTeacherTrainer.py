import tensorflow as tf
from collections.abc import Callable, Sequence

from .util import downsample_masks

import json,os
IMAGE_SHAPE = json.loads(os.environ['IMAGE_SHAPE'])


class MeanTeacherTrainer:

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        loss_supervised_fn: Callable[[tf.Tensor,tf.Tensor],tf.Tensor],
        ds_weights: Sequence[float],
        loss_consistency_fn: Callable[[tf.Tensor,tf.Tensor],tf.Tensor],
        student_optimizer: tf.keras.mixed_precision.LossScaleOptimizer
    ):
        self.student = student
        self.teacher = teacher
        self.loss_supervised_fn = loss_supervised_fn
        self.ds_weights = tf.constant(ds_weights, dtype='float32')
        self.loss_consistency_fn = loss_consistency_fn
        self.student_optimizer = student_optimizer

        # computing output shapes for dataset indices
        output_shapes = student.compute_output_shape([None]+IMAGE_SHAPE+[2])
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
        tf.TensorSpec(shape=[None]+IMAGE_SHAPE+[1], dtype='float32'),
        tf.TensorSpec(shape=[None]+IMAGE_SHAPE+[2], dtype='float32'),
        tf.TensorSpec(shape=[None]+IMAGE_SHAPE+[2], dtype='float32'),
        tf.TensorSpec(shape=(), dtype='float32'),
        tf.TensorSpec(shape=(), dtype='float32'),
    ), jit_compile=True)
    def train_step(
        self, mris_source, masks_source, mris_target, mris_target_v2, consistency_weight, ema_decay
    ):
        down_masks = [downsample_masks(masks_source, self.w_indices[i], self.h_indices[i], self.d_indices[i]) for i in range(self.n_ds)]
        preds_teacher = self.teacher(mris_target_v2, training=True)
        with tf.GradientTape() as tape:
            preds_sup = self.student(mris_source, training=True)
            loss1 = self.loss_supervised_fn(masks_source,preds_sup[0])
            losses_aux = [self.loss_supervised_fn(down_masks[i], preds_sup[i+1]) for i in range(self.n_ds)]
            loss_supervised = loss1*self.ds_weights[0] + tf.math.reduce_sum(tf.stack(losses_aux)*self.ds_weights[1:])

            preds_unsup = self.student(mris_target, training=True)[0]
            loss_consistency = self.loss_consistency_fn(preds_teacher, preds_unsup)

            loss_global = loss_supervised + consistency_weight*loss_consistency
            loss_ = self.student_optimizer.get_scaled_loss(loss_global)

        grads = tape.gradient(loss_, self.student.trainable_variables)
        grads = self.student_optimizer.get_unscaled_gradients(grads)
        self.student_optimizer.apply_gradients(zip(grads,self.student.trainable_variables))
        for t_var,s_var in zip(self.teacher.weights, self.student.weights):
            t_var.assign(t_var*ema_decay + s_var*(1-ema_decay))
        return loss_supervised,loss_consistency