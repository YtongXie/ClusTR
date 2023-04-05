## ClusTR: Exploring Efficient Self-attention via Clustering for Vision Transformers

This is the official pytorch implementation of the ClusTR:<br />

**Paper: [ClusTR: Exploring Efficient Self-attention via Clustering for Vision Transformers](https://arxiv.org/pdf/2208.13138.pdf).**

The code is developed on top of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [pvt](https://github.com/whai362/PVT).


## Requirements
CUDA 11.3<br />
Python 3.10<br /> 
Pytorch 1.11<br />
Torchvision 0.12.0<br />
mmcv 1.4.8<br />
timm <br />


## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training
To train ClusTR-b1 on ImageNet on a single node with 8 gpus for 300 epochs run:

```
sh dist_train.sh configs/ClusTR/ClusTR_b1.py 8 --data-path ImageNet --batch-size 128 --epochs 300
```

## Evaluation
To evaluate a pre-trained ClusTR-b1 on ImageNet val with a single GPU run:
```
sh dist_train.sh configs/ClusTR/ClusTR_b1.py 1 --data-path ImageNet --resume /path/to/checkpoint_file --eval
```



If you use this code for a paper please cite:
```
@article{xie2022clustr,
  title={ClusTR: Exploring Efficient Self-attention via Clustering for Vision Transformers},
  author={Xie, Yutong and Zhang, Jianpeng and Xia, Yong and Hengel, Anton van den and Wu, Qi},
  journal={arXiv preprint arXiv:2208.13138},
  year={2022}
}
```
