import numpy as np
import math


def conv_iter(image, kernel, out_array, bias, out_i, out_j, ksize, global_i, global_j):
    in_chan = kernel.shape[0]
    for i in range(ksize):
        for j in range(ksize):
            for in_ch in range(in_chan):
                out_array[out_i, out_j] += image[in_ch, global_i + i, global_j + j] * kernel[in_ch, i, j]
    out_array[out_i, out_j] += bias


def conv2d(image: np.ndarray, kernel: np.ndarray, bias: np.ndarray, stride: int) -> np.ndarray:
    '''Naive implementation of convolution 2D
       ARGS:
        image : np.ndarray - input image in CHW format
        kernel : np.ndarray - convolution kernel in CDHW format
        bias : np.ndarray - convolution kernel bias of shape D
        stride : int convolution stride
    '''
    in_chan, out_chan, ksize, _ = kernel.shape
    _, h, w = image.shape
    out_array = np.zeros((out_chan, math.ceil((h - ksize + 1) / stride), math.ceil((w - ksize + 1) / stride)))
    out_i = out_j = 0
    for i in range(0, h - ksize + 1, stride):
        for j in range(0, w - ksize + 1, stride):
            for out_ch in range(out_chan):
                conv_iter(image, kernel[:, out_ch,...], out_array[out_ch, ...], bias[out_ch], out_i, out_j, ksize, i, j)
            out_j += 1

        out_j = 0
        out_i += 1
    return out_array
    
                