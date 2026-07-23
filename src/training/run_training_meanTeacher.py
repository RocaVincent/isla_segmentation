from collections.abc import Sequence
from typing import Literal
from time import time
import json,os
import tensorflow as tf
import numpy as np
import logging
from datetime import timedelta

from .input_pipeline.tf_dataset import training_dataset_meanTeacher
from ..model_architectures.unet import UNet, InputShape, UNetType
from ..model_architectures.serialize_unet import save_model_config
from .loss_functions import sum_dice_bce, asym_unified_focal_loss, mse
from .trainers.meanTeacherTrainer import MeanTeacherTrainer

def train_meanTeacher(
    tfrecord_paths_source: Sequence[str],
    tfrecord_paths_target: Sequence[str],
    unet_type: Literal['StdUNet','ResUNet'],
    loss_supervised_str: Literal['DiceBCE','UFL'],
    output_dir: str,
    n_epochs: int,
    batch_size_source: int,
    batch_size_target: int,
    steps_per_epoch: int, # must be computed by user
    verbose: int,
    std_output: bool
) -> None:
    t_start = time()
    if os.path.isdir(output_dir): print(f"WARNING: The specified output {output_dir} directory already exists.")
    else: os.mkdir(output_dir)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    ### DATASET INITIALIZATION ###
    dataset = training_dataset_meanTeacher(
        tfrecord_paths_source=tfrecord_paths_source,
        tfrecord_paths_target=tfrecord_paths_target,
        batch_size_source=batch_size_source,
        batch_size_target=batch_size_target,
        cache=False,
        buffer_size_source=400,
        buffer_size_target=400,
        compression=True
    )

    ### MODEL INSTANTIATIONS ###
    input_shape = json.loads(os.environ['IMAGE_SHAPE'])
    input_shape = InputShape(tuple(input_shape))
    student = UNet(
        input_shape=input_shape,
        unet_type=UNetType(unet_type),
        deep_supervision=True,
        dropout_deepLayers=True
    )
    teacher = UNet(
        input_shape=input_shape,
        unet_type=UNetType(unet_type),
        deep_supervision=False,
        dropout_deepLayers=False
    )
    weights = student.get_weights()
    teacher.set_weights([weights[i] for i in range(len(teacher.get_weights()))])

    ### LOSS FUNCTIONS ###
    match loss_supervised_str:
        case 'DiceBCE':
            loss_supervised_fn = sum_dice_bce
            init_lr = 0.001
            consistency_weight_end = 5
        case 'UFL':
            loss_supervised_fn = asym_unified_focal_loss()
            init_lr = 0.005
            consistency_weight_end = 2
        case _: raise Exception(f"Unknown loss {loss_supervised_str}")
    loss_consistency_fn = mse

    ### STUDENT OPTIMIZER ###
    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=n_epochs*steps_per_epoch,
        end_learning_rate=1e-6,
        power=1
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.99,
        nesterov=True
    )
    student_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=131072, dynamic_growth_steps=200)

    ### TRAINER ###
    trainer = MeanTeacherTrainer(
        student=student,
        teacher=teacher,
        loss_supervised_fn=loss_supervised_fn,
        ds_weights=(0.53333333, 0.26666667, 0.13333333, 0.06666667),
        loss_consistency_fn=loss_consistency_fn,
        student_optimizer=student_optimizer
    )

    ### MEAN TEACHER HYPERPARAMETERS ###
    n_epochs_rampUp = n_epochs//5
    def get_consistency_weight(epoch):
        return consistency_weight_end * np.exp(-5*np.square(1-min(epoch,n_epochs_rampUp)/n_epochs_rampUp))
    
    EMA_DECAY_INIT = 0.99
    EMA_DECAY_END = 0.999
    deltaEma_decay = EMA_DECAY_END-EMA_DECAY_INIT
    deltaEpoch_ema_decay = deltaEma_decay/n_epochs
    def get_ema_decay(epoch):
        return EMA_DECAY_INIT + epoch*deltaEpoch_ema_decay


    ### LOG ###
    handlers=[logging.FileHandler(output_dir+'/train.log', mode='w')]
    if std_output: handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=handlers
    )

    ### MAIN LOOP ###
    losses_supervised = np.empty(n_epochs)
    losses_consistency = np.empty(n_epochs)
    losses_supervised_epoch = np.empty(steps_per_epoch)
    losses_consistency_epoch = np.empty(steps_per_epoch)
    for epoch in range(n_epochs):
        consistency_weight = get_consistency_weight(epoch+1)
        ema_decay = get_ema_decay(epoch+1)
        for step, ((mris_source,masks_source),(mris_target,mris_target_v2)) in enumerate(dataset):
            losses_supervised_epoch[step],losses_consistency_epoch[step] = trainer.train_step(
                mris_source=mris_source,
                masks_source=masks_source,
                mris_target=mris_target,
                mris_target_v2=mris_target_v2,
                consistency_weight=consistency_weight,
                ema_decay=ema_decay
            )
            if verbose==2 and std_output:
                print(f"End step {step+1}/{steps_per_epoch}, loss_supervised = {losses_supervised_epoch[step]:.4f}, loss_consistency = {losses_consistency_epoch[step]:.4f}",
                     end='\r')
        losses_supervised[epoch] = losses_supervised_epoch.mean()
        losses_consistency[epoch] = losses_consistency_epoch.mean()
        if verbose>=1:
            logging.info(f"End epoch {epoch+1}/{n_epochs} -> loss_supervised = {losses_supervised[epoch]:.4f}, losses_consistency = {losses_consistency[epoch]:.4f}")

    ### SAVING and final prompt ###
    save_model_config(student, f"{output_dir}/student_config.json")
    save_model_config(teacher, f"{output_dir}/teacher_config.json")
    student.save_weights(f"{output_dir}/final_student_weights.h5")
    teacher.save_weights(f"{output_dir}/final_teacher_weights.h5")
    d = {
        'losses_supervised':list(losses_supervised),
        'losses_consistency':list(losses_consistency)
    }
    with open(f"{output_dir}/stats_train.json", 'w') as f:
        f.write(json.dumps(d, indent=4))
    if verbose>=1:
        duration = time()-t_start
        logging.info(f"End of training (execution time : {timedelta(seconds=int(duration))}), model weights and training losses are saved in {output_dir}")