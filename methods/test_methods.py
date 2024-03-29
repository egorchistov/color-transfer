from functools import partial

import pytest
from skimage.util import img_as_float
from skimage.data import chelsea
from skimage.metrics import peak_signal_noise_ratio as psnr

from methods.linear import color_transfer_between_images
from methods.linear import color_transfer_in_correlated_color_space
from methods.linear import monge_kantorovitch_color_transfer
from methods.iterative import iterative_distribution_transfer
from methods.iterative import automated_color_grading


@pytest.mark.parametrize("method", [
    color_transfer_between_images,
    color_transfer_in_correlated_color_space,
    partial(monge_kantorovitch_color_transfer, decomposition="cholesky"),
    partial(monge_kantorovitch_color_transfer, decomposition="sqrt"),
    partial(monge_kantorovitch_color_transfer, decomposition="MK"),
    iterative_distribution_transfer,
    automated_color_grading,
])
def test_method(method):
    """Create artificial distortions which should be easily corrected by any method"""
    image_true = img_as_float(chelsea())

    image_test = 1.15 * image_true + 0.2
    image_corr = method(image_test, image_true)

    psnr_before = psnr(image_true, image_test, data_range=1)
    psnr_after = psnr(image_true, image_corr, data_range=1)

    psnr_gain = psnr_after - psnr_before

    print(f"PSNR gain: {psnr_gain:.3f}, PSNR after: {psnr_after:.3f}")

    assert psnr_gain > 10
