
# DEVR:Distillation Efficient Vison-RWKV


This repository contains PyTorch evaluation code, training code and pretrained models for the DEVR:


# Model Zoo

We provide baseline DEVR models pretrained on ImageNet 2012.

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| DEVR-tiny | 76.6 | 91.1 | 6.3M | [model]() |
| DEVR-small | 81.7 | 96.1 | 20.9M| coming soon |
| DEVR-base | 83.0 | 97.3 | 88.9M | coming soon |

The models are also available via torch hub.
Before using it, make sure you have the pytorch-image-models package [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman) installed. Note that our work relies of the augmentations proposed in this library. 

To load DEVR-base with pretrained weights on ImageNet simply do:

```python
import torch
# check you have the right version of timm
import timm
assert timm.__version__ == "0.3.2"

# now load it with torchhub
model = torch.hub.load('', 'devr_base_patch16_224', pretrained=True)
```

Additionnally, we provide a [Colab notebook]() which goes over the steps needed to perform inference with DEVR.

# Usage

First, clone the repository locally:
```
git clone https://github.com/
```
Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Evaluation
To evaluate a pre-trained DEVR-base on ImageNet val with a single GPU run:
```
python main.py --eval --resume devr_t.pth --data-path /path/to/imagenet
```
This should give
```
* Acc@1 76.662 Acc@5 93.168 loss 1.169
```



## Training
To train DEVR-small and DEVR-tiny with hard distillation using a ConvneXt as teacher,on ImageNet on a single node with single gpus for 300 epochs run:

DEVR-small
```
python --model devr_small --batch-size 256 --distillation-type hard --teacher-model convnext_small --data-path /path/to/imagenet --output_dir /path/to/save
```

DEVR-tiny
```
python --model devr_tiny --batch-size 256 --distillation-type hard --teacher-model convnext_tiny --data-path /path/to/imagenet --output_dir /path/to/save
```





# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


