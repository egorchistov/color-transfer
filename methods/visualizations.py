import math

import torch
import numpy as np
from kornia.metrics import ssim
from kornia.color import rgb_to_lab


def chess_mix(x, y, size=25):
    height, width = x.shape[-2:]
    result = torch.zeros_like(x)

    for i in range(math.ceil(height / size)):
        for j in range(math.ceil(width / size)):
            block = np.s_[...,
                          i * size: min((i + 1) * size, height),
                          j * size: min((j + 1) * size, width)]
            source = x if (i + j) % 2 == 0 else y
            result[block] = source[block]

    return result


def minmaxscale(x, dim=(-1, -2)):
    min_values = x.amin(dim=dim, keepdim=True)
    max_values = x.amax(dim=dim, keepdim=True)

    return (x - min_values) / (max_values - min_values)


def rgbmse(x, y):
    error = torch.zeros_like(x)
    m = torch.square(x - y).mean(dim=1)
    error[:, 0] = minmaxscale(m)

    return error


def labmse(x, y):
    error = torch.zeros_like(x)
    m = rgb_to_lab(torch.square(x - y))[:, (0, 1, 2)].mean(dim=1)
    error[:, 0] = minmaxscale(m)

    return error


def abmse(x, y):
    error = torch.zeros_like(x)
    m = rgb_to_lab(torch.square(x - y))[:, (1, 2)].mean(dim=1)
    error[:, 0] = minmaxscale(m)

    return error


def rgbssim(x, y):
    error = torch.zeros_like(x)
    m = 0.5 - ssim(x, y, window_size=11).mean(dim=1) / 2
    error[:, 0] = minmaxscale(m)

    return error
