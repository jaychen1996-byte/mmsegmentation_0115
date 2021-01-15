from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from skimage import io

"""
Image_20200925100338349.bmp
../configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes_custom_binary.py
../tools/work_dirs/deeplabv3_r50-d8_512x1024_40k_cityscapes_custom_binary/iter_2000.pth
--device
cuda:0
--palette
cityscapes_custom_binary
"""


def main():
    parser = ArgumentParser()
    # parser.add_argument('--img', default="Image_20200925100338349.bmp", help='Image file')
    parser.add_argument('--img', default="star.png", help='Image file')
    # parser.add_argument('--img', default="demo.png", help='Image file')
    parser.add_argument('--config',
                        # default="../configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes_custom_binary.py",
                        default="../configs/danet/danet_r50-d8_512x1024_40k_cityscapes_custom.py",
                        # default="../configs/deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes.py",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        # default="../tools/work_dirs/deeplabv3_r50-d8_512x1024_40k_cityscapes_custom_binary/iter_200.pth",
                        default="../tools/work_dirs/danet_r50-d8_512x1024_40k_cityscapes_custom/iter_4000.pth",
                        # default="../checkpoints/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes_20200908_005644-cf9ce186.pth",
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes_custom',
        # default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)

    # io.imsave("result.png", result[0])
    # io.imshow(result[0])
    # io.show()
    # show the results
    show_result_pyplot(model, args.img, result, get_palette(args.palette))

    """
    python demo/image_demo.py demo/demo.jpg configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --palette cityscapes
    demo.png ../configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py ../checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --palette cityscapes
    demo.png ../configs/deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes.py ../checkpoints/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes_20200908_005644-cf9ce186.pth --device cuda:0 --palette cityscapes
    demo.png ../configs/point_rend/pointrend_r101_512x512_160k_ade20k.py ../checkpoints/pointrend_r101_512x512_160k_ade20k_20200808_030852-8834902a.pth --device cuda:0 --palette cityscapes
    """


if __name__ == '__main__':
    main()
