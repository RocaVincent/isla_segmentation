import tensorflow as tf

@tf.function(input_signature=(
    tf.TensorSpec(shape=[None]+IMAGE_SHAPE+[1], dtype='float32'),
    tf.TensorSpec(shape=[None], dtype='int32'),
    tf.TensorSpec(shape=[None], dtype='int32'),
    tf.TensorSpec(shape=[None], dtype='int32')
))
def downsample_masks(masks, w_idx, h_idx, d_idx):
    masks = tf.gather(masks, w_idx, axis=1)
    masks = tf.gather(masks, h_idx, axis=2)
    masks = tf.gather(masks, d_idx, axis=3)    
    return masks