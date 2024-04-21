import numpy as np
import time
import torch
from torch.nn.functional import conv2d as torch_conv

from conv2d import conv2d
from im2col import conv_im2col


def test_conv2d():
    image = np.random.randn(3, 256, 256)
    kernel = np.random.randn(3, 5, 3, 3)
    bias = np.random.randn(5)
    for stride in range(1, 6):
        print('Testing with stride:', stride)
        torch_conv_out = torch_conv(torch.tensor(image), torch.tensor(kernel.transpose(1, 0, 2, 3)),
                                     torch.tensor(bias), stride).numpy()
        out_conv = conv2d(image, kernel, bias, stride)
        print('Is output equal to expected:', np.allclose(torch_conv_out, out_conv))
        print('=' * 80)


def test_equal_conv2d_and_im2col():
    '''Check equality of results naive conv2d implementation and im2col'''
    image = np.random.randn(3, 256, 256)
    kernel = np.random.randn(3, 5, 3, 3)
    bias = np.random.randn(5)
    for stride in range(1, 6):

        print('Testing with stride:', stride)

        # warm up
        out_conv = conv2d(image, kernel, bias, stride)
        out_im2col = conv_im2col(image, kernel, bias, stride)

        print('Time measuring and comparison of classic conv2d and im2col:')
        start = time.time()
        out_conv = conv2d(image, kernel, bias, stride)
        print('Naive conv2d:', time.time() - start, 'seconds')

        start = time.time()
        out_im2col = conv_im2col(image, kernel, bias, stride)
        print('Im2Col', time.time() - start, 'seconds')

        print('Is outputs equal:', np.allclose(out_conv, out_im2col))
        print('\n' + '=' * 80 + '\n')


if __name__=='__main__':
    test_conv2d()
    print('\n' * 2 + 'Testing im2col:' + '\n')
    test_equal_conv2d_and_im2col()
