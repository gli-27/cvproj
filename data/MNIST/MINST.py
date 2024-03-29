# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:06:24 2019

@author: gli
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


DATA_PATH = './'
train_images = 'train-images-idx3-ubyte'
train_labels = 'train-labels-idx1-ubyte'
test_images = 't10k-images-idx3-ubyte'
test_labels = 't10k-labels-idx1-ubyte'

def load_mnist_image(path, fileName, type='train'):
    filePath = os.path.join(path, fileName)
    fp = open(filePath, 'rb')
    buf = fp.read()
    index = 0
    magic, num, rows, cols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for image in range(0, num):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im, dtype='uint8')
        im = im.reshape(28, 28)
        im = Image.fromarray(im)
        if (type == 'train'):
            isExists = os.path.exists('./train')
            if not isExists:
                os.mkdir('./train')
            im.save('./train/train_%s.bmp' % image, 'bmp')
        elif (type == 'test'):
            isExists = os.path.exists('./test')
            if not isExists:
                os.mkdir('./test')
            im.save('./test/test_%s.bmp' % image, 'bmp')

def load_mnist_label(path, fileName, type='train'):
    filePath = os.path.join(path, fileName)
    fp = open(filePath, 'rb')
    buf = fp.read()
    index = 0
    magic, num = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    Labels = np.zeros(num)

    for i in range(num):
        Labels[i] = np.array(struct.unpack_from('>B', buf, index))
        index += struct.calcsize('>B')

    if (type == 'train'):
        np.savetxt('./train_labels.csv', Labels, fmt='%i', delimiter=',')
    if (type == 'test'):
        np.savetext('./test_labels.csv', Labels, fmt='%i', delimiter=',')

    return Labels

if __name__ == '__main__':
    load_mnist_image(DATA_PATH, train_images, 'train')
    load_mnist_label(DATA_PATH, train_labels, 'train')
    load_mnist_image(DATA_PATH, test_images, 'test')
    load_mnist_label(DATA_PATH, test_labels, 'test')
