import os
from argparse import ArgumentParser
from style_transfer_utils import *

######################################################################

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--style-image', type=str,
                        dest='style_image', help='Reference image for image/video styling',
                        metavar='STYLE_IMAGE', required=True)

    parser.add_argument('--content', type=str,
                        dest='content', help='Image or video to be transformed',
                        metavar='CONTENT', required=True)

    return parser

######################################################################

if __name__ == '__main__':

    parser = build_parser()

    try:
        options = parser.parse_args()
    except:
        _ = input()
        raise SystemExit

    style_path = options.style_image
    content_path = options.content

    try:
        assert os.path.exists(style_path), 'The specified style image could not be found'
        assert os.path.exists(content_path), 'The scpecified content file could not be found'
    except AssertionError as error:
        print(error)
        _ = input()
        raise SystemExit

    image_stylizer = Stylizer(style_path)

    contentIsVideo = False
    if content_path[-4:] == '.mp4':
        contentIsVideo = True
    
    output = image_stylizer.stylize(content_path)

    if contentIsVideo:
        show_video(output)
    else:
        img = PIL.Image.fromarray(output)
        save_path = os.path.dirname(content_path) + '\\' + os.path.basename(content_path).split('.')[0] + '_stylized.jpg'
        img.save(save_path)
        imshow(output, 'Stylized Image')
        plt.show()