_base_ = [
    '../_base_/models/deeplabv3_r50-d8_custom_single.py', '../_base_/datasets/cityscapes_custom_binary.py',
    '../_base_/default_runtime_deeplabv3_r50-d8_512x1024.py', '../_base_/schedules/schedule_1k.py'
]
