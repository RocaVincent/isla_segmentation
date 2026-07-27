from collections.abc import Sequence
from typing import Literal
from time import time
import json,os
import tensorflow as tf
import numpy as np
import logging
from datetime import timedelta


IMAGE_SHAPE = json.loads(os.environ['IMAGE_SHAPE'])

def train_fullySupervised(
    tfrecord_paths: Sequence[str],
    unet_type: Literal['StdUNet','ResUNet'],
    loss_str: Literal['DiceBCE','UFL'],
    output_dir: str,
    n_epochs: int,
    batch_size: int,
    steps_per_epoch: int, # must be computed by user
    verbose: int,
    std_output: bool
) -> None:
    t_start = time()
    if os.path.isdir(output_dir): print(f"WARNING: The specified output {output_dir} directory already exists.")
    else: os.mkdir(output_dir)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus)==0: raise Exception("No GPUs detected.")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    from ...model_architectures.unet import UNet, InputShape, UNetType
    from ...model_architectures.serialize_unet import save_model_config
    from ..input_pipeline.tf_dataset import training_dataset_supervised
    from ..loss_functions import sum_dice_bce, asym_unified_focal_loss
    from ..trainers.fullySupervisedTrainer import FullySupervisedTrainer

    ### DATASET INITIALIZATION ###
    dataset = training_dataset_supervised(
        tfrecord_paths=tfrecord_paths,
        batch_size=batch_size,
        cache=False,
        buffer_size=800,
        compression=True,
    )

    ### MODEL INSTANTIATION ###
    input_shape = json.loads(os.environ['IMAGE_SHAPE'])
    model = UNet(
        input_shape=InputShape(tuple(input_shape)),
        unet_type=UNetType(unet_type),
        deep_supervision=True,
        dropout_deepLayers=True
    )

    ### LOSS FUNCTION ###
    match loss_str:
        case 'DiceBCE':
            loss_fn = sum_dice_bce
            init_lr = 0.001
        case 'UFL':
            loss_fn = asym_unified_focal_loss()
            init_lr = 0.005
        case _: raise Exception(f"Unknow loss {loss_str}")

    ### OPTIMIZER ###
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
    ls_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, initial_scale=131072, dynamic_growth_steps=200)

    ### TRAINER ###
    trainer = FullySupervisedTrainer(
        model=model,
        loss_fn=loss_fn,
        ds_weights=(0.53333333, 0.26666667, 0.13333333, 0.06666667),
        ls_optimizer=ls_optimizer
    )

    ### LOG ###
    handlers=[logging.FileHandler(output_dir+'/train.log', mode='w')]
    if std_output: handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=handlers
    )
        

    ### MAIN LOOP ###
    losses_train = np.empty(n_epochs)
    losses_epoch = np.empty(steps_per_epoch)
    for epoch in range(n_epochs):
        for step,(mris,masks) in enumerate(dataset):
            losses_epoch[step] = trainer.train_step(mris,masks)
            if verbose==2 and std_output:
                print(f"End step {step+1}/{steps_per_epoch}, loss = {losses_epoch[step]:.4f}", end='\r')
        losses_train[epoch] = losses_epoch.mean()
        if verbose>=1:
            logging.info(f"End epoch {epoch+1}/{n_epochs} -> training loss = {losses_train[epoch]:.4f}")

    ### SAVING and final prompt ###
    save_model_config(model, f"{output_dir}/config.json")
    model.save_weights(f"{output_dir}/final_weights.h5")
    d = {
    'losses_train': list(losses_train)
    }
    with open(f"{output_dir}/stats_train.json", 'w') as f:
        f.write(json.dumps(d, indent=4))
    if verbose>=1:
        duration = time()-t_start
        logging.info(f"End of training (execution time : {timedelta(seconds=int(duration))}), model weights and training losses are saved in {output_dir}")