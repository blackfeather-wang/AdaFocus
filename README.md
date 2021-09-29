# AdaFocus (ICCV-2021 Oral)

This repo contains the official code and pre-trained models for AdaFocus.

- [Adaptive Focus for Efficient Video Recognition](http://arxiv.org/abs/2105.03245)

## Reference
If you find our code or paper useful for your research, please cite:
```
@InProceedings{Wang_2021_ICCV,
author = {Wang, Yulin and Chen, Zhaoxi and Jiang, Haojun and Song, Shiji and Han, Yizeng and Huang, Gao},
title = {Adaptive Focus for Efficient Video Recognition},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
}
```

## Introduction

In this paper, we explore the spatial redundancy in video recognition with the aim to improve the computational efficiency. It is observed that the most informative region in each frame of a video is usually a small image patch, which shifts smoothly across frames. Therefore, we model the patch localization problem as a sequential decision task, and propose a reinforcement learning based approach for efficient spatially adaptive video recognition (AdaFocus). In specific, a light-weighted ConvNet is first adopted to quickly process the full video sequence, whose features are used by a recurrent policy network to localize the most task-relevant regions. Then the selected patches are inferred by a high-capacity network for the final prediction. During offline inference, once the informative patch sequence has been generated, the bulk of computation can be done in parallel, and is efficient on modern GPU devices. In addition, we demonstrate that the proposed method can be easily extended by further considering the temporal redundancy, e.g., dynamically skipping less valuable frames. Extensive experiments on five benchmark datasets, i.e., ActivityNet, FCVID, Mini-Kinetics, Something-Something V1\&V2, demonstrate that our method is significantly more efficient than the competitive baselines.


<p align="center">
    <img src="./figure/intro.png" width= "475">
</p>


## Result

- ActivityNet

<p align="center">
    <img src="./figure/actnet.png" width= "550">
</p>


- Something-Something V1&V2

<p align="center">
    <img src="./figure/sthsth.png" width= "900">
</p>


- Visualization

<p align="center">
    <img src="./figure/visualization.png" width= "900">
</p>

## Requirements
- python 3.8
- pytorch 1.7.0
- torchvision 0.8.0
- [hydra](https://hydra.cc/docs/intro/) 1.1.0

## Datasets
1. Please get train/test splits file for each dataset from [Google Drive](https://drive.google.com/drive/folders/1L41U4mczsrnwiSx3KiY57BblrRE5fjnU?usp=sharing) and put them in `PATH_TO_DATASET`.
2. Download videos from following links, or contact the corresponding authors for the access. Save them to `PATH_TO_DATASET/videos`
- [ActivityNet-v1.3](http://activity-net.org/download.html) 
- [FCVID](https://drive.google.com/drive/folders/1cPSc3neTQwvtSPiVcjVZrj0RvXrKY5xj)
- [Mini-Kinetics](https://deepmind.com/research/open-source/kinetics). Please download [Kinetics 400](https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz), for Mini-Kinetics used in our paper, you need to use the train/val splits file.
3. Extract frames using [ops/video_jpg.py](ops/video_jpg.py), the frames will be saved to `PATH_TO_DATASET/frames`. Minor modifications on file path are needed when extracting frames from different dataset.



## Pre-trained Models

Please download pretrained weights and checkpoints from [Google Drive](https://drive.google.com/drive/folders/1v5UnucCr2CjmH41HEJePPI2WtDO8SSsp?usp=sharing).

- globalcnn.pth.tar: pretrained weights for global CNN (MobileNet-v2).
- localcnn.pth.tar: pretrained weights for local CNN (ResNet-50).
- 128checkpoint.pth.tar: checkpoint of stage 1 for patch size 128x128.
- 160checkpoint.pth.tar: checkpoint of stage 1 for patch size 160x128.
- 192checkpoint.pth.tar: checkpoint of stage 1 for patch size 192x128.
 
## Training

- Here we take training model with patch size 128x128 on ActivityNet dataset for example.
- All logs and checkpoints will be saved in the directory: `./outputs/YYYY-MM-DD/HH-MM-SS`
- Note that we store a set of default paramenter in [conf/default.yaml](conf/default.yaml) which can override through command line. You can also use your own config files.

- Before training, please initialize Global CNN and Local CNN by fine-tuning the ImageNet pre-trained models in Pytorch using the following command:

for Global CNN:
```
CUDA_VISIBLE_DEVICES=0,1 python main_dist.py dataset=actnet data_dir=PATH_TO_DATASET train_stage=0 batch_size=64 workers=8 dropout=0.8 lr_type=cos backbone_lr=0.01 epochs=15 dist_url=tcp://127.0.0.1:8857 random_patch=true patch_size=128 glance_size=224 eval_freq=5 consensus=gru hidden_dim=1024 pretrain_glancer=true
```
for Local CNN:
```
CUDA_VISIBLE_DEVICES=0,1 python main_dist.py dataset=actnet data_dir=PATH_TO_DATASET train_stage=0 batch_size=64 workers=8 dropout=0.8 lr_type=cos backbone_lr=0.01 epochs=15 dist_url=tcp://127.0.0.1:8857 random_patch=true patch_size=128 glance_size=224 eval_freq=5 consensus=gru hidden_dim=1024 pretrain_glancer=false
```

- Training stage 1, pretrained weights for Global CNN and Local CNN are required:
```
CUDA_VISIBLE_DEVICES=0,1 python main_dist.py dataset=actnet data_dir=PATH_TO_DATASET train_stage=1 batch_size=64 workers=8 dropout=0.8 lr_type=cos backbone_lr=0.0005 fc_lr=0.05 epochs=50 dist_url=tcp://127.0.0.1:8857 random_patch=true patch_size=128 glance_size=224 eval_freq=5 consensus=gru hidden_dim=1024 pretrained_glancer=PATH_TO_CHECKPOINTS pretrained_focuser=PATH_TO_CHECKPOINTS
```

- Training stage 2, a stage-1 checkpoint is required:
```
CUDA_VISIBLE_DEVICES=0 python main_dist.py dataset=actnet data_dir=PATH_TO_DATASET train_stage=2 batch_size=64 workers=8 dropout=0.8 lr_type=cos backbone_lr=0.0005 fc_lr=0.05 epochs=50 random_patch=false patch_size=128 glance_size=224 action_dim=49 eval_freq=5 consensus=gru hidden_dim=1024 resume=PATH_TO_CHECKPOINTS multiprocessing_distributed=false distributed=false
```

- Training stage 3, a stage-2 checkpoint is required:
```
CUDA_VISIBLE_DEVICES=0,1 python main_dist.py dataset=actnet data_dir=PATH_TO_DATASET train_stage=3 batch_size=64 workers=8 dropout=0.8 lr_type=cos backbone_lr=0.0005 fc_lr=0.005 epochs=10 random_patch=false patch_size=128 glance_size=224 action_dim=49 eval_freq=5 consensus=gru hidden_dim=1024 resume=PATH_TO_CHECKPOINTS multiprocessing_distributed=false distributed=false
```

## Contact
If you have any question, feel free to contact the authors or raise an issue. 
Yulin Wang: wang-yl19@mails.tsinghua.edu.cn.

## Acknowledgement
We use implementation of MobileNet-v2 and ResNet from [Pytorch source code](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html). We also borrow some codes for dataset preparation from [AR-Net](https://github.com/mengyuest/AR-Net#dataset-preparation) and PPO from [here](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py).
