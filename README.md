# basic-style-transfer

This is a simple implementation of neural style transfer based on the Tensorflow Neural Style Transfer tutorial.

https://www.tensorflow.org/tutorials/generative/style_transfer

The main script is in style_transfer.py using the following command line arguements:
- '--style-image': The image used as a style reference 
- '--content': The image or video to be transformed

**Example Usage**

```
python style_transfer.py --style-image /path/to/styleimage --content /path/to/content
```

Stylised images and videos will be saved in the jpeg and mp4 formats respectively.

The package prerequisites to successfully run the script can be installed using the requirements.txt file.
```
python -m pip install -r requirements.txt
```
