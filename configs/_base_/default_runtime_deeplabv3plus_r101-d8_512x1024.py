# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/jaychen/Desktop/PycharmProjects/2020.11.05/mmsegmentation/checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
