_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_custom.py',
    '../_base_/datasets/cityscapes_custom.py', '../_base_/default_runtime_deeplabv3plus_r101_d16-mg124_512x1024.py',
    '../_base_/schedules/schedule_1k.py'
]
