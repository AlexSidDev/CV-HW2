import numpy as np
import time

from conv2d import conv2d
from im2col import conv_im2col


def test_conv():
    image = np.concatenate([np.full((1, 5, 5), i) for i in range(3, 0, -1)], axis=0)
    kernel = np.concatenate([np.full((3, 1, 3, 3), i) for i in range(1, 4)], axis=1)
    stride = 1
    out = conv2d(image, kernel, stride)
    print(out)


def test_equal_conv2d_and_im2col():
    '''Check equality of results naive conv2d implementation and im2col'''
    image = np.concatenate([np.full((1, 256, 256), i) for i in range(3, 0, -1)], axis=0)
    kernel = np.concatenate([np.full((3, 1, 3, 3), i) for i in range(1, 4)], axis=1)
    stride = 1

    # warm up
    out_conv = conv2d(image, kernel, stride)
    out_im2col = conv_im2col(image, kernel, stride)
    
    print('Time measuring and comparison of classic conv2d and im2col:')
    start = time.time()
    out_conv = conv2d(image, kernel, stride)
    print('Naive conv2d:', time.time() - start, 'seconds')

    start = time.time()
    out_im2col = conv_im2col(image, kernel, stride)
    print('Im2Col', time.time() - start, 'seconds')

    print('Is outputs equal:', np.allclose(out_conv, out_im2col))