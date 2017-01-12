import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from PIL import Image


#Generates gifs
def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy
  
  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)
  
  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration,verbose=False)

#Function loads images from list of files. bw_bool is True when the source images are originally greyscale and 4:3
#Flip determines whether images should be flipped 
def loadImages(data,bw_bool,flip):
    images = []
    images_bw = []
    if bw_bool == False:
        for myFile in data:
            img = Image.open(myFile)
            bw = np.max(img,2)
            bw = np.stack([bw,bw,bw],2)
            bw[:,:40,:] = 0
            bw[:,-40:,:] = 0
            if flip == False:
                images.append(np.array(img))
                images_bw.append(bw)
            else:
                img_flip = np.fliplr(img)
                images.append(img_flip)
                bw_flip = np.fliplr(bw)
                images_bw.append(bw_flip)
        images = np.array(images)
        images = images.astype('float32')
        images = images / 256
        images_bw = np.array(images_bw)
        images_bw = images_bw.astype('float32')
        images_bw = images_bw / 256
        return images,images_bw
    else:
        for myFile in data:
            img = Image.open(myFile)
            bw = img.resize((196,144))
            bw = np.max(bw,2)
            bw = np.stack([bw,bw,bw],2)
            bw_w = np.zeros([144,256,3])
            bw_w[:,30:-30,:] = bw
            bw_w[:,:40,:] = 0
            bw_w[:,-40:,:] = 0
        images.append(bw_w)
        images = np.array(images)
        images = images.astype('float32')
        images = images / 256
        return images

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
    
#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1],3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = image

    return img
