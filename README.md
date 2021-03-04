#  PolypSegmentation

##  Introduction

This repository is forkd from [ACSNet](https://github.com/ReaFly/ACSNet). I have extended the code to support more models and make it easier to use.
On top of the original code, I added some function: progress bar display, log logging, multi-scale training for all networks, etc.

Currently supported models:
* Unet
* ACSNet
* PraNet


##  Requirements

* torch
* torchvision 
* tqdm
* opencv
* scipy
* skimage
* PIL
* numpy

##  Usage

####  1. Training

```bash
# To train PraNet, for example.
python train.py  --config pranet --exp_name pranet --gpu_ids 0
# You can resume traning by:
python train.py  --config pranet --continue_train
```

More network and training configurations need to be modified in `configs`.

####  2. Inference

```bash
python test.py  --config pranet 
```






