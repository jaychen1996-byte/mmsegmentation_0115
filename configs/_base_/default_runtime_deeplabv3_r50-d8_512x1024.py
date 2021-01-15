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
load_from = '/home/jaychen/Desktop/PycharmProjects/2020.11.05/mmsegmentation/checkpoints/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
