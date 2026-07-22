import tensorflow as tf

import json,os
IMAGE_SHAPE = json.loads(os.environ['IMAGE_SHAPE'])


@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
    tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
))
def dice_loss(y_true, y_pred):
    SMOOTH = 1.0
    num = 2*tf.math.reduce_sum(y_true*y_pred, axis=(1,2,3,4)) + SMOOTH
    den = tf.math.reduce_sum(y_true+y_pred, axis=(1,2,3,4)) + SMOOTH
    return tf.math.reduce_mean(-num/den)

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
    tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
))
def binary_crossentropy(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred, axis=(0,1,2,3,4), from_logits=False)

@tf.function(input_signature=(
    tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
    tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
))
def sum_dice_bce(y_true, y_pred):
    return dice_loss(y_true,y_pred) + binary_crossentropy(y_true, y_pred)



def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    
    
    def asymmetric_focal_loss(delta=0.7, gamma=2.):
        """For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
        """
        def loss_function(y_true, y_pred):
            loss_fg = -y_true*tf.math.log(y_pred)
            loss_bg = -(1-y_true)*tf.math.pow(y_pred,gamma)*tf.math.log(1-y_pred)
            voxel_loss = delta*loss_fg + (1-delta)*loss_bg
            return tf.math.reduce_mean(voxel_loss)

        return loss_function
    
    def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
        """This is the implementation for binary segmentation.
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            focal parameter controls degree of down-weighting of easy examples, by default 0.75
        """
        def loss_function(y_true, y_pred):
            tp = tf.math.reduce_sum(y_true*y_pred, axis=(1,2,3,4))
            tn = tf.math.reduce_sum((1-y_true)*(1-y_pred), axis=(1,2,3,4))

            fp = tf.math.reduce_sum((1-y_true)*y_pred, axis=(1,2,3,4))
            fn = tf.math.reduce_sum(y_true*(1-y_pred), axis=(1,2,3,4))

            loss_fg = 1-tp/(tp+delta*fp+(1-delta)*fn)
            loss_bg = 1-tn/(tn+delta*fn+(1-delta)*fp)

            batch_loss = tf.math.pow(loss_fg, 1-gamma) + loss_bg
            return tf.math.reduce_mean(batch_loss/2) # division par 2 pas dans le papier mais semble dans le code

        return loss_function
    
    ftl_fn = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)
    fl_fn = asymmetric_focal_loss(delta=delta, gamma=gamma)
    
    EPS_UFL = 1e-7
    @tf.function(input_signature=(
        tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
        tf.TensorSpec(shape=IMAGE_SHAPE+[1], dtype='float32'),
    ))
    def loss_function(y_true,y_pred):
        y_pred = tf.clip_by_value(y_pred, EPS_UFL, 1-EPS_UFL)
        asymmetric_ftl = ftl_fn(y_true,y_pred)
        asymmetric_fl = fl_fn(y_true,y_pred)
        return weight * asymmetric_ftl + (1-weight) * asymmetric_fl

    return loss_function