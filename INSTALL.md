# Anaconda environment

## Environment creation

Your environment should have Python 3.10.13.

```shell
conda create -n isla python=3.10.13
conda activate isla
```


## TensorFlow installation

TensorFlow 2.15.0 is required:

```shell
pip install \
    tensorflow==2.15.0 \
    tensorflow-graphics==2021.12.3
```

GPU support is recommended for faster inference, and required for training:

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
    nvidia-nccl-cu12==2.16.5 \
    nvidia-nvjitlink-cu12==12.2.140
```


## Pytorch installation

Pytorch 2.4.0 for CPU is required for SynthStrip.

```shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```


## Other packages

```shell
pip install \
  pandas==2.2.2 \
  nibabel==5.2.1 \
  antspyx==0.6.3 \
  surfa==0.6.3
```