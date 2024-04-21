import math
import numpy as np


def conv_im2col(image: np.ndarray, kernel: np.ndarray, bias: np.ndarray, stride: int):
    '''Implementation of convolution 2D using Im2Col algorithm
       ARGS:
        image : np.ndarray - input image in CHW format
        kernel : np.ndarray - convolution kernel in CDHW format
        bias : np.ndarray - convolution kernel bias of shape D
        stride : int convolution stride
    '''
    in_chan, out_chan, ksize, _ = kernel.shape
    _, h, w = image.shape

    out_shape = (out_chan, math.ceil((h - ksize + 1) / stride), math.ceil((w - ksize + 1) / stride))
    reshaped_image = np.zeros((in_chan * ksize * ksize, out_shape[1] * out_shape[2]))

    reshaped_kernel = kernel.transpose(1, 0, 2, 3).reshape(out_chan, -1)
    reshaped_kernel = np.concatenate([reshaped_kernel, bias[:, None]], axis=1)
    n = 0
    for i in range(0, h - ksize + 1, stride):
        for j in range(0, w - ksize + 1, stride):
            reshaped_image[:, n] = image[:, i: i + ksize, j: j + ksize].reshape(-1)
            n += 1
    
    out_array = reshaped_kernel @ np.concatenate([reshaped_image, np.ones((1, out_shape[1] * out_shape[2]))], axis=0)

    return out_array.reshape(*out_shape)