## [BASNet: Burned Area Segmentation Network for Real-Time Detection of Damage Maps in Remote Sensing Images](https://ieeexplore.ieee.org/document/9852471)

By  Weihao Bo, Jie Liu, Xijian Fan, Tardi Tjahjadi, Qiaolin Ye, Liyong Fu

The PyTorch code for IEEE TGRS paper "BASNet: Burned Area Segmentation Network for Real-Time Detection of Damage Maps in Remote Sensing Images"  

## Introduction

### BASNet framework

![BASNet](./img/BASNet.png)

## Usage

### 1. Clone the resposity

```bash
git clone https://github.com/weihao-bo/BASNet.git
cd BASNet
```

### 2. Install Requirements

```
Python 3.6
Pytorch 1.4+
OpenCV 4.0
Numpy
TensorboardX
Apex
```

### 3. Datasets

The burned area dataset used in this paper is not available due to copyright issues. The format of the dataset is the same as the regular SOD dataset like [DUTS](http://saliencydetection.net/duts/).

Download the datasets and unzip them into `data` folder.

### 4. Training

### 5. Evaluation

### 6. Results

## Citation

If you find this work is helpful in your research, please cite:

```
@ARTICLE{9852471,
  author={Bo, Weihao and Liu, Jie and Fan, Xijian and Tjahjadi, Tardi and Ye, Qiaolin and Fu, Liyong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={BASNet: Burned Area Segmentation Network for Real-Time Detection of Damage Maps in Remote Sensing Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2022.3197647}}
```

## Acknowledgement

This project is based on the implementation of [CTDNet](https://github.com/iCVTEAM/CTDNet).

Thanks to [CTDNet](https://github.com/iCVTEAM/CTDNet), [forest-fire-damage-mapping](https://github.com/daitranskku/forest-fire-damage-mapping) and [TRACER](https://github.com/Karel911/TRACER) for their help in this work.

