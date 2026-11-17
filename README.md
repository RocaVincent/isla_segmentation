# Acute ischemic stroke segmentation with ISLA

This repository provides the code and model weights to use the ISLA model for segmentation of ischemic stroke lesions. ISLA has been trained to segment acute ischemic stroke lesion from trace diffusion-weighted images and apparent diffusion coefficient maps. Details of the method and its validation experiments are provided in [our paper](https://doi.org/10.48550/arXiv.2601.08732).


## Installation instructions

Installation instructions are available in [INSTALL.md](INSTALL.md).


## ISLA inference

Run the following command to perform inference with ISLA:

```shell
python -m src.inference.main \
    --cases-csv my_inputs.csv \
    --configs-weights-csv isla_configs_weights/configs_weights.csv
```

Replace *my_inputs.csv* with the path to a CSV file containing three columns specifying the input and output filepaths: *dwi*, *adc* and *output*.

The inference script implements the complete ISLA pipeline, including the following preprocessing steps:

1. Skull-stripping with Synthstrip
2. Bias field correction with N4ITK
3. Reorientation
4. Resampling
5. Padding/cropping
6. Intensity normalization


If you train your own models (see [ISLA training](#isla-training)), replace *isla_config_weights/configs_weights.csv* with the path to your own CSV file specifying the model configurations and corresponding weight files.

Run the following command to display all available options:

```shell
python -m src.inference.main -h
```


## ISLA training

### Data Preparation

To train ISLA on your own data, your images must be preprocessed and converted in the TFRecord format. Use the `create_tfrecords` function in [this script](src/training/tfrecords/tfrecords_supervised.py).

For ISLA-MT, which supports unsupervised domain adaptation, you can additionally use the `create_tfrecords` function in [this script](src/training/tfrecords/tfrecords_unsupervised.py).


### Fully-supervised training

Use `python -m src.training.mains.main_fullySupervised` for classical fully-supervised training.

Options:

  - --image-shape IMAGE_SHAPE : '160,192,80' or '256,256,48'. Has to match with the --tfrecord-paths argument
                        
  - --tfrecord-paths TFRECORD_PATHS : Path of a file containing comma-separated GZIP-compressed tfrecord paths for training. Each tfrecord entry has two keys : 'mris', 'lesion'.
  
  - --unet-type UNET_TYPE : 'StdUNet' or 'ResUNet'
  
  - --loss LOSS : 'DiceBCE' or 'UFL'
  
  - --output-dir OUTPUT_DIR
  - --n-samples N_SAMPLES : User has to provide the number of entries in --tfrecords-path.

Run the following command to display all available options:

```shell
python -m src.training.mains.main_fullySupervised -h
```


### Mean Teacher training

Use `python -m src.training.mains.main_meanTeacher` for Mean Teacher training for unsupervised domain adaptation.

Options:

  - --image-shape IMAGE_SHAPE : MRI input shape. '160,192,80' or '256,256,48'. Has to match with the --tfrecord-paths-source and --tfrecord-paths-target argument.
  
  - --tfrecord-paths-source TFRECORD_PATHS_SOURCE : Path of a file containing comma-separated GZIP-compressed tfrecord paths for supervised training. Each tfrecord entry has two keys : 'mris', 'lesion'.
  
  - --tfrecord-paths-target TFRECORD_PATHS_TARGET : Path of a file containing comma-separated GZIP-compressed tfrecord paths for unsupervised training. Each tfrecord entry has one key : 'mris'.
  
  - --unet-type UNET_TYPE : 'StdUNet' or 'ResUNet'
  
  - --loss-supervised LOSS_SUPERVISED: 'DiceBCE' or 'UFL'
  
  - --output-dir OUTPUT_DIR
  
  - --n-samples-source N_SAMPLES_SOURCE : User has to provide the number of entries in --tfrecords-path-source.

Run the following command to display all available options:

```shell
python -m src.training.mains.main_meanTeacher -h
```