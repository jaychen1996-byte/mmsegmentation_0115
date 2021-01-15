_base_ = [
    '../_base_/models/deeplabv3_r50-d8_custom.py', '../_base_/datasets/cityscapes_custom.py',
    '../_base_/default_runtime_deeplabv3_r50-d8_512x1024.py', '../_base_/schedules/schedule_1k.py'
]
