checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)
hybrid_decoder = dict(type='SequenceAttentionDecoder')
position_decoder = dict(type='PositionAttentionDecoder')
model = dict(
    type='RobustScannerTwoStage',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='ChannelReductionEncoder', in_channels=512, out_channels=128),
    decoder=dict(
        type='RobustScannerDecoderTwoStage',
        dim_input=512,
        dim_model=128,
        hybrid_decoder=dict(type='SequenceAttentionDecoder'),
        position_decoder=dict(type='PositionAttentionDecoder')),
    loss=dict(type='SARLoss'),
    label_convertor=dict(
        type='AttnConvertor', dict_type='DICT90', with_unknown=True),
    max_seq_len=30)
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=160,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'valid_ratio'
                ])
        ])
]
dataset_type = 'OCRDataset'
train_prefix = 'F:/OCR/dataset/'
train_img_prefix1 = 'F:/OCR/dataset/SynthText/synthtext/SynthText_patch_horizontal'
train_img_prefix2 = 'F:/OCR/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px'
train_ann_file1 = 'F:/OCR/dataset/SynthText/label.lmdb'
train_ann_file2 = 'F:/OCR/dataset/mjsynth/label.lmdb'
train1 = dict(
    type='OCRDataset',
    img_prefix='F:/OCR/dataset/SynthText/synthtext/SynthText_patch_horizontal',
    ann_file='F:/OCR/dataset/SynthText/label.lmdb',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train2 = dict(
    type='OCRDataset',
    img_prefix='F:/OCR/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px',
    ann_file='F:/OCR/dataset/mjsynth/label.lmdb',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
test_prefix = 'F:/OCR/dataset/'
test_img_prefix1 = 'F:/OCR/dataset/IIIT5K/'
test_img_prefix2 = 'F:/OCR/dataset/svt/'
test_img_prefix3 = 'F:/OCR/dataset/ic13/'
test_img_prefix4 = 'F:/OCR/dataset/ic15/'
test_img_prefix5 = 'F:/OCR/dataset/svtp/'
test_img_prefix6 = 'F:/OCR/dataset/CUTE80/'
test_ann_file1 = 'F:/OCR/dataset/IIIT5K/test_label.txt'
test_ann_file2 = 'F:/OCR/dataset/svt/test_label.txt'
test_ann_file3 = 'F:/OCR/dataset/ic13/test_label_1015.txt'
test_ann_file4 = 'F:/OCR/dataset/ic15/test_label.txt'
test_ann_file5 = 'F:/OCR/dataset/svtp/test_label.txt'
test_ann_file6 = 'F:/OCR/dataset/CUTE80/lable.txt'
test1 = dict(
    type='OCRDataset',
    img_prefix='F:/OCR/dataset/IIIT5K/',
    ann_file='F:/OCR/dataset/IIIT5K/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test2 = dict(
    type='OCRDataset',
    img_prefix='F:/OCR/dataset/svt/',
    ann_file='F:/OCR/dataset/svt/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test3 = dict(
    type='OCRDataset',
    img_prefix='F:/OCR/dataset/ic13/',
    ann_file='F:/OCR/dataset/ic13/test_label_1015.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test4 = dict(
    type='OCRDataset',
    img_prefix='F:/OCR/dataset/ic15/',
    ann_file='F:/OCR/dataset/ic15/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test5 = dict(
    type='OCRDataset',
    img_prefix='F:/OCR/dataset/svtp/',
    ann_file='F:/OCR/dataset/svtp/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test6 = dict(
    type='OCRDataset',
    img_prefix='F:/OCR/dataset/CUTE80/',
    ann_file='F:/OCR/dataset/CUTE80/lable.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
data = dict(
    samples_per_gpu=270,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                'F:/OCR/dataset/SynthText/synthtext/SynthText_patch_horizontal',
                ann_file='F:/OCR/dataset/SynthText/label.lmdb',
                loader=dict(
                    type='LmdbLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px',
                ann_file='F:/OCR/dataset/mjsynth/label.lmdb',
                loader=dict(
                    type='LmdbLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=160,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'text',
                    'valid_ratio'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/IIIT5K/',
                ann_file='F:/OCR/dataset/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/svt/',
                ann_file='F:/OCR/dataset/svt/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/ic13/',
                ann_file='F:/OCR/dataset/ic13/test_label_1015.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/ic15/',
                ann_file='F:/OCR/dataset/ic15/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/svtp/',
                ann_file='F:/OCR/dataset/svtp/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/CUTE80/',
                ann_file='F:/OCR/dataset/CUTE80/lable.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=48,
                        min_width=48,
                        max_width=160,
                        keep_aspect_ratio=True,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'resize_shape',
                            'valid_ratio'
                        ])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/IIIT5K/',
                ann_file='F:/OCR/dataset/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/svt/',
                ann_file='F:/OCR/dataset/svt/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/ic13/',
                ann_file='F:/OCR/dataset/ic13/test_label_1015.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/ic15/',
                ann_file='F:/OCR/dataset/ic15/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/svtp/',
                ann_file='F:/OCR/dataset/svtp/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='F:/OCR/dataset/CUTE80/',
                ann_file='F:/OCR/dataset/CUTE80/lable.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=48,
                        min_width=48,
                        max_width=160,
                        keep_aspect_ratio=True,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'resize_shape',
                            'valid_ratio'
                        ])
                ])
        ]))
evaluation = dict(interval=1, metric='acc')
work_dir = './roubust_two_stage/'
gpu_ids = range(0, 2)
