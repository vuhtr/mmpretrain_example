# MMPretrained Quick Start

## Installation

```bash
conda create -n classify python=3.9 -y
conda activate classify
bash scripts/install.sh
```

## Dataset Preparation

Your dataset folder should be in this format:

```bash
dataset_name
    |-- train/
        |-- class1/
            |-- *.jpg
        |-- class2/
            |-- *.jpg
        |-- ...
    |-- val/
        |-- class1/
            |-- *.jpg
        |-- class2/
            |-- *.jpg
        |-- ...
    |-- test/ (similar to train/ and val/, if needed)
```

## Configurations

Official documents: https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html

Example config files can be found at `mmpretrain/configs/example`.

Let's consider the file `./mmpretrain/configs/example/mobilenetv2_384.py`.

### Model settings

These lines are important for model settings:

Line 2:

```python
_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',  <--- Choose the file correponding to your desire backbone
    ...
]
```
Line 11 to 24:

- Get the checkpoint link from model zoo: https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html

```python
num_classes = NUMBER_OF_CLASSES (ex: real and fake -> 2 classes)

model = dict(
    backbone=dict(
        frozen_stages=-1,   # not freeze backbone
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',   <-- Get from the model zoo
            prefix='backbone',
        )),
    head=dict(
        num_classes=num_classes,
    )
)
```

### Data pipeline settings

This part is quite straightforward (from line 30 to 98), these lines are important to consider:
- Train and validation batch size should be modify as well

```python

input_size = 384
data_root = 'ABSOLUTE_PATH_TO_YOUR_DATASET_FOLDER, e.g dataset_name folder in the above'
metainfo = {
    'classes': ['cls1', 'cls2'],
}

data_preprocessor = dict(
    num_classes=num_classes,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,    # should always be True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=input_size), # or RandomResizedCrop
    ###################################
    # You can change these augmentation pipeline as you want
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1)
        ],
        prob=0.8
    ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    ###################################
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=input_size),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=1,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [
    dict(type='Accuracy'),
    dict(type='SingleLabelMetric', items=['precision', 'recall'])
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator

```

### Optimizer settings

From line 103 to 134, we setup the optimizer and learning rate scheduler. These lines are important to consider:

```python
max_epoch = 100
warm_up_epoch = 5

# optimizer
optim_wrapper = dict(
    # optimizer=dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
    optimizer=dict(type='Adam', lr=0.001)
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=warm_up_epoch,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch - warm_up_epoch,
        by_epoch=True,
        begin=warm_up_epoch,
        end=max_epoch,
        eta_min=1e-6,
        convert_to_iter_based=True)
]
```

## Training

Single GPU training:

```bash
cd mmpretrain

python tools/train.py <path_to_your_config_file>
```

Multi-GPU training:

```bash
cd mmpretrain

CUDA_VISIBLE_DEVICES=id1,id2,... bash tool/dis_train.sh <path_to_your_config_file> <number_of_gpus>
```

## Evaluation

To evaluate our trained model on the validation set:

```bash
cd mmpretrain

python tools/test.py <path_to_your_config_file> <path_to_checkpoint_file>
```

