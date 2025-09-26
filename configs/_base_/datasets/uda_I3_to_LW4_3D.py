# Obtained from: https://github.com/lhoyer/HRDA
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[0, ], std=[1, ], to_rgb=True)
crop_size = (256, 256)
gta_train_pipeline = [
    dict(type='LoadImageFromFile',from_3d=True),
    dict(type='LoadAnnotations',from_3d=True),
    dict(type='Resize', img_scale=(256, 256),from_3d=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75,from_3d=True),
    dict(type='RandomFlip3D', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize',from_3d=True, **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255,from_3d=True),
    dict(type='DefaultFormatBundle',from_3d=True),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'sam_pseudo_label']), #<------- ADDED
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile',from_3d=True),
    dict(type='LoadAnnotations',from_3d=True),
    dict(type='Resize', img_scale=(256, 256),from_3d=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75,from_3d=True),
    dict(type='RandomFlip3D', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize',from_3d=True, **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255,from_3d=True),
    dict(type='DefaultFormatBundle',from_3d=True),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'sam_pseudo_label']), #<------- ADDED
]
test_pipeline = [
    dict(type='LoadImageFromFile',from_3d=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(256, 256),from_3d=True),
            dict(type='RandomFlip3D', prob=0.5),
            dict(type='Normalize',from_3d=True, **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='I3toLW4Dataset',
            data_root='data/I3/patch_dataset/',
            img_dir='images',
            ann_dir='labels',
            pseudo_label_dir='sam', #<------- ADDED
            pipeline=gta_train_pipeline),
        target=dict(
            type='I3toLW4Dataset',
            data_root='data/LW4/patch_dataset/',
            img_dir='images',
            ann_dir='labels',
            pseudo_label_dir='sam', #<------- ADDED
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='I3toLW4Dataset',
        data_root='data/LW4/patch_dataset/',
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline),
    test=dict(
        type='I3toLW4Dataset',
        data_root='data/LW4/patch_dataset/',
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline))
