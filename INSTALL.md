# Anaconda environment

## Environment creation

Start by creating an Anaconda environment:

```shell
conda create -n isla python=3.10.13
conda activate isla
```


## TensorFlow installation

Then install TensorFlow 2.15.0 with GPU support (recommended for inference and mandatory for training):

```shell
pip install tensorflow==2.15.0 --no-deps
```

For GPU support (recommended if available), you need to install the CUDA/CUDNN packages TensorFlow 2.15.0 was build against:

```shell
pip install \
  nvidia-cublas-cu12==12.2.5.6 \
  nvidia-cuda-cupti-cu12==12.2.142 \
  nvidia-cuda-nvcc-cu12==12.2.140 \
  nvidia-cuda-nvrtc-cu12==12.2.140 \
  nvidia-cuda-runtime-cu12==12.2.140 \
  nvidia-cudnn-cu12==8.9.4.25 \
  nvidia-cufft-cu12==11.0.8.103 \
  nvidia-curand-cu12==10.3.3.141 \
  nvidia-cusolver-cu12==11.5.2.141 \
  nvidia-cusparse-cu12==12.1.2.141 \
  nvidia-nccl-cu12==2.16.5
```

Then you need to reinstall the correct non-CUDA dependencies:

```shell
pip install \
  keras==2.15.0 \
  ml-dtypes==0.2.0 \
  numpy==1.26.4 \
  protobuf==3.20.3 \
  tensorboard==2.15.2 \
  wrapt==1.14.1 \
  tensorflow-estimator==2.15.0 \
  tensorflow-io-gcs-filesystem==0.36.0 \
  gast==0.5.4 \
  flatbuffers==23.5.26 \
  google-pasta==0.2.0 \
  termcolor==2.4.0 \
  opt-einsum==3.3.0 \
  astunparse==1.6.3 \
  libclang==16.0.6 \
  grpcio==1.60.0 \
  h5py==3.10.0 \
  google-auth==2.27.0 \
  google-auth-oauthlib==1.0.0 \
  typing-extensions==4.16.0
```

## Other packages

```shell
pip install \
  pandas==2.2.2 \
  tensorflow-graphics==2021.12.3 \
  nibabel==5.2.1
```