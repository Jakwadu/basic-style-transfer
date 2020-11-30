import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import functools
import cv2
from tqdm import trange

######################################################################

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

######################################################################

def tensor_to_image(tensor, numpy=False):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
    tensor = tensor[0]
    if numpy:
        return tensor
    else:
        return PIL.Image.fromarray(tensor)

######################################################################

def show_video(path):
    generatedVideo = cv2.VideoCapture(path)
    filename = path.split("\\")[-1]
    if not generatedVideo.isOpened():
        print("Failed to open {}".format(filename))
    else:
        while generatedVideo.isOpened():
            retVal, frame = generatedVideo.read()

            if not retVal:
                generatedVideo.release()
                break
            
            cv2.imshow(filename, frame)
            
            if cv2.waitKey(20) and 0xFF == ord('q'):
                generatedVideo.release()
                cv2.destroyAllWindows()
                break
        
        generatedVideo.release()
        cv2.destroyAllWindows()

######################################################################

class Stylizer():
    def __init__(self, style):
        self.max_dim = 512
        self.style_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        self.style = self.load_img(style)[0]

    def stylize(self, content):
        extension = content[-4:]
        if extension == '.jpg':
            content, original_shape = self.load_img(content)
            stylized_image = self.style_model(tf.constant(content), tf.constant(self.style))[0]
            stylized_image = tensor_to_image(stylized_image, numpy=True)
            stylized_image = cv2.resize(stylized_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LANCZOS4)
            return stylized_image
        elif extension == '.mp4':
            return self.generate_frames(content)

    def load_img(self, path_to_img):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img, old_shape = self.resize(img, return_shape=True)

        return img, old_shape

    def resize(self, img, return_shape=False):
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(input=img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = self.max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]

        if return_shape:
            return img, shape
        else:
            return img

    def generate_frames(self, content):
        tmp = content.split('\\')
        directory, filename = tmp[:-1], tmp[-1][:-4]
        directory = "\\".join(directory)
        stylized_video = directory + "\\" + "stylized_" + filename + ".mp4"
        
        video = cv2.VideoCapture(content)
        retVal, frame = video.read()

        if not retVal:
            print('Failed to read video')
            pass
        
        h, w = frame.shape[0], frame.shape[1]
        video = cv2.VideoCapture(content)
        videoLen = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frameRate = int(video.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(stylized_video, fourcc, frameRate, (w, h), True)
        video = cv2.VideoCapture(content)

        for idx in trange(videoLen, desc='Stylizing video frames'):
            retVal, frame = video.read()
            
            if not retVal:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.resize(frame)
            
            stylized_frame = self.style_model(tf.constant(frame, dtype=tf.float32), tf.constant(self.style))[0]
            stylized_frame = tensor_to_image(stylized_frame, numpy=True)
            
            if np.ndim(stylized_frame) > 3:
                stylized_frame = np.squeeze(stylized_frame, axis=0)
            
            stylized_frame = tf.image.resize(stylized_frame, (h,w))
            stylized_frame = cv2.cvtColor(np.array(stylized_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            
            if not writer.isOpened():
                writer.open(stylized_video, fourcc, frameRate, (w, h), True)            
            
            if writer.isOpened():
                writer.write(stylized_frame)
            
            if idx == videoLen - 1:
                print('File write complete!')

        writer.release()

        print('Saved stylized video to {}'.format(stylized_video))
        print('Frame dimensions: {}x{}'.format(w,h))
        print('Frame rate: {}'.format(frameRate))

        return stylized_video