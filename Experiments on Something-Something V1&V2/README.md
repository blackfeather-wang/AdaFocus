# Experiments on Something-Something V1&V2

## Requirements
- python 3.8
- pytorch 1.8.0
- torchvision 0.8.0
- [hydra](https://hydra.cc/docs/intro/) 1.1.0

## Datasets
Please follow the instruction of [TSM](https://github.com/mit-han-lab/temporal-shift-module#data-preparation) to process the Something-Something V1/V2 dataset.

## Pre-trained Models on Something-Something-V1 (V2)

Please download pre-trained weights and checkpoints from [Google Drive](https://drive.google.com/drive/folders/1QgIjU6FLT3RZbAGAVutgOPuOOOtBPpFb?usp=sharing).

- Something-Something-V1 (V2)
    - mobilenetv2_segment8.pth.tar: pre-trained weights for global CNN (MobileNet-v2).
    - resnet50_segment12.pth.tar: pre-trained weights for local CNN (ResNet-50).
    - 144x144.pth.tar: checkpoint to reproduce the result in paper with patch size 144x144.
    - 160x160.pth.tar: checkpoint to reproduce the result in paper with patch size 160x160.
    - 176x176.pth.tar: checkpoint to reproduce the result in paper with patch size 176x176.

## Training

- Here we take training the model with patch size 144x144 on Something-Something-V1 dataset for example.
- All logs and checkpoints will be saved in the directory: `./outputs/YYYY-MM-DD/HH-MM-SS`
- Note that we store a set of default hyper-parameters in [conf/default.yaml](conf/default.yaml) which can be overrided through command line. You can also use your own config files.

- Before training, please initialize Global CNN and Local CNN by fine-tuning the ImageNet pre-trained models in Pytorch using the following command:

For Global CNN: please use the [TSM code](https://github.com/mit-han-lab/temporal-shift-module#data-preparation) and use the following command:
```
 python main.py something RGB \
       --arch mobilenetv2 --num_segments 8 \
       --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
       --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres --npb
```
For Local CNN: please use the [TSM code](https://github.com/mit-han-lab/temporal-shift-module#data-preparation) and use the following command:
```
 python main.py something RGB \
       --arch resnet50 --num_segments 12 \
       --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
       --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres --npb
```

- Training stage 1, we provide the command in train_stage1.sh. The pre-trained weights for Global CNN and Local CNN are required, so you should set the pretrained_glancer and pretrained_focuser arguments both right in train_stage1.sh first and then run it:
```
cd sth-sth
bash train_stage1.sh
```

- Training stage 2, we provide the command in train_stage2.sh. A stage-1 checkpoint is required, so you should set the pretrained argument right in train_stage2.sh and then run it:
```
bash train_stage2.sh
```

- Training stage 3, we provide the command in train_stage2.sh. A stage-1 checkpoint is required, so you should set the pretrained_s2 argument right in train_stage3.sh and then run it:
```
bash train_stage3.sh
```


## Evaluate Pre-trained Models
- Here we take evaluating model with patch size 144x144 on Something-Something-V1 dataset for example. 
 
    We provide the command in evaluate.sh. The pre-trained weights is required, so you should set the resume and patch_size arguments both right in evaluate.sh first and then run it:

```
bash evaluate.sh
```

## Acknowledgement
We use the official implementation of [temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module) and PPO from [here](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py).
