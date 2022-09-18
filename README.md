## [BASNet: Burned Area Segmentation Network for Real-Time Detection of Damage Maps in Remote Sensing Images](https://ieeexplore.ieee.org/document/9852471)

By  Weihao Bo, Jie Liu, Xijian Fan, Tardi Tjahjadi, Qiaolin Ye, Liyong Fu

### **Abstract:**

Since remote sensing images of post-fire vegetation are characterized by high resolution, multiple interferences, and high similarities between the background and the target area, it is difficult for existing methods to detect and segment the burned area in these images with sufficient speed and accuracy. In this article, we apply salient object detection (SOD) to burned area segmentation (BAS), the first time this has been done, and propose an efficient burned area segmentation network (BASNet) to improve the performance of unmanned aerial vehicle (UAV) high-resolution image segmentation. BASNet comprises positioning module and refinement module. The positioning module efficiently extracts high-level semantic features and general contextual information via global average pooling layer and convolutional block (CB) to determine the coarse location of the salient region. The refinement module adopts the CB attention module to effectively discriminate the spatial location of objects. In addition, to effectively combine edge information with spatial location information in the lower layer of the network and the high-level semantic information in the deeper layer, we design the residual fusion module to perform feature fusion by level to obtain the prediction results of the network. Extensive experiments on two UAV datasets collected from Chongli in China and Andong in South Korea, demonstrate that our proposed BASNet significantly outperforms the state-of-the-art SOD methods quantitatively and qualitatively. BASNet also achieves a promising prediction speed for processing high-resolution UAV images, thus providing wide-ranging applicability in post-disaster monitoring and management.

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

```
cd src
python train.py
```

- We use [ResNet-18](https://download.pytorch.org/models/resnet18-5c106cde.pth) as the backbone network.
- We train our model on our dataset captured in ChongLi District and the dataset provided by the work [Damage-Map Estimation Using UAV Images and Deep Learning Algorithms for Disaster Management System](https://www.mdpi.com/2072-4292/12/24/4169) captured in Andong City.
- After training, the result models will be saved in `out` folder. 

### 5. Testing

```
cd src
python train.py
```

- After testing, the result saliency maps will be saved in `eval` folder.

### 6. Evaluation

```
    cd eval
    matlab main
```

- We evaluate the performance of our BASNet with MATLAB code.

### 7.Results

- Chongli District dataset, China

  | Method | F-measure |  MAE  | E-measure |
  | :----: | :-------: | :---: | :-------: |
  | BASNet |   0.772   | 0.010 |   0.767   |

- Andong City dataset, South Korea

  | Method | F-measure |  MAE  | E-measure |
  | :----: | :-------: | :---: | :-------: |
  | BASNet |   0.615   | 0.016 |   0.657   |

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

