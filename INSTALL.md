# Cloning repository

Clone the repository and navigate to its root directory:

```shell
git clone https://github.com/RocaVincent/isla_segmentation.git
cd isla_segmentation
```


# Model weights and configuration file

Download and extract the model configurations and pretrained weights:

```shell
wget https://github.com/RocaVincent/isla_segmentation/releases/download/v1.0.0/isla_configs_weights.zip
unzip isla_configs_weights.zip
```

Download the Synthstrip weights:

```shell
wget https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt
```



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


## Required for training only

```shell
pip install \
    elasticdeform==0.5.1 \
    scikit-image==0.24.0
```