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
# load_from = '/home/jaychen/Desktop/PycharmProjects/2020.11.05/mmsegmentation/checkpoints/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes_20200908_005644-cf9ce186.pth'
load_from = None
resume_from = '/home/jaychen/Desktop/PycharmProjects/2020.11.05/mmsegmentation/tools/work_dirs/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes_custom/iter_2000.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
