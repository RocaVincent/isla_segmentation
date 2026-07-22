import tensorflow as tf

from .data_augmentation import augmentation_fns

import json,os
IMAGE_SHAPE = json.loads(os.environ['IMAGE_SHAPE'])


supervised_augmentation, unsupervised_augmentation = augmentation_fns(
  flip=True,
  shift=True,
  rotation=True,
  elastic=True,
  scaling=True,
  gaussNoise=True,
  gaussBlur=True,
  brightness=True,
  contrast=True,
  lowRes=True,
  gamma=True
)

def training_dataset_supervised(
    tfrecord_paths: list[str],
    batch_size: int,
    cache: bool,
    buffer_size: int,
    compression: bool = True
) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(tfrecord_paths, num_parallel_reads=tf.data.AUTOTUNE, compression_type='GZIP' if compression else None)
    keys_to_feature = {'mris':   tf.io.FixedLenFeature(shape=IMAGE_SHAPE+[2], dtype='float32'),
                       'lesion': tf.io.FixedLenFeature(shape=IMAGE_SHAPE+[1], dtype='float32')}
    def parse_function(record):
        d = tf.io.parse_single_example(record, keys_to_feature)
        return d['mris'], d['lesion']

    if cache:
        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        dataset = dataset.cache().shuffle(buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.map(supervised_augmentation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    else:
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.map(lambda record: supervised_augmentation(*parse_function(record)), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        
    return dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(1)
    

def training_dataset_meanTeacher(
    tfrecord_paths_source: list[str],
    tfrecord_paths_target: list[str],
    batch_size_source: int,
    batch_size_target: int,
    cache: bool,
    buffer_size_source: int,
    buffer_size_target: int,
    compression: bool = True
) -> tf.data.Dataset:
    dataset_source = tf.data.TFRecordDataset(tfrecord_paths_source, num_parallel_reads=tf.data.AUTOTUNE,
                                             compression_type='GZIP' if compression else None)
    dataset_target = tf.data.TFRecordDataset(tfrecord_paths_target, num_parallel_reads=tf.data.AUTOTUNE,
                                             compression_type='GZIP' if compression else None)
    keys_to_feature_source = {'mris':   tf.io.FixedLenFeature(shape=IMAGE_SHAPE+[2], dtype='float32'),
                              'lesion': tf.io.FixedLenFeature(shape=IMAGE_SHAPE+[1], dtype='float32')}
    keys_to_feature_target = {'mris':   tf.io.FixedLenFeature(shape=IMAGE_SHAPE+[2], dtype='float32')}
    def parse_source(record):
        d = tf.io.parse_single_example(record, keys_to_feature_source)
        return d['mris'], d['lesion']
    def parse_target(record):
        return tf.io.parse_single_example(record, keys_to_feature_target)['mris']

    if cache:
        dataset_source = dataset_source.map(parse_source, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        dataset_source = dataset_source.cache().shuffle(buffer_size_source, reshuffle_each_iteration=True)
        dataset_source = dataset_source.map(supervised_augmentation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        dataset_target = dataset_target.map(parse_target, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        dataset_target = dataset_target.cache().shuffle(buffer_size_target, reshuffle_each_iteration=True)
        dataset_target = dataset_target.map(unsupervised_augmentation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    else:
        dataset_source = dataset_source.shuffle(buffer_size_source, reshuffle_each_iteration=True)
        dataset_source = dataset_source.map(lambda record: supervised_augmentation(*parse_source(record)),
                                            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        dataset_target = dataset_target.shuffle(buffer_size_target, reshuffle_each_iteration=True)
        dataset_target = dataset_target.map(lambda record: unsupervised_augmentation(parse_target(record)),
                                            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    dataset_source = dataset_source.batch(batch_size_source, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    dataset_target = dataset_target.batch(batch_size_target, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE,
                                          deterministic=False).repeat() # repeat to make epochs run the entire source 
    return tf.data.Dataset.zip(dataset_source, dataset_target).prefetch(1)