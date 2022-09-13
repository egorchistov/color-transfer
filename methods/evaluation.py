"""Distortion Based Evaluation

We use undistored image pairs and then introduce color mismatch,
applying different color modification operators to one of the images:

* No modification
* Brightness (-30, -20, -10, 10, 20, 30)
* Contrast (-30, -20, -10, 10, 20, 30)
* Exposure (-3, -2, -1, 1, 2, 3)
* Hue (-30, -20, -10, 10, 20, 30)
* Saturation (-30, -20, -10, 10, 20, 30)
* Value (-30, -20, -10, 10, 20, 30)

For evaluation we use MSE metric in LAB color space and SSIM.
We calculate average metric gain for each distortion type and
also average metric values.
"""

from pathlib import Path
from collections import defaultdict
from functools import partial

from matplotlib import pyplot as plt
from skimage import io, img_as_ubyte, img_as_float
from skimage.color import rgb2lab
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import albumentations.augmentations.functional as F
from tqdm import tqdm

from methods.linear import color_transfer_between_images


distortions = {
    "undist": lambda x: x,
    "bri-30": partial(F.brightness_contrast_adjust, alpha=1, beta=-0.30),
    "bri-20": partial(F.brightness_contrast_adjust, alpha=1, beta=-0.20),
    "bri-10": partial(F.brightness_contrast_adjust, alpha=1, beta=-0.10),
    "bri+10": partial(F.brightness_contrast_adjust, alpha=1, beta=+0.10),
    "bri+20": partial(F.brightness_contrast_adjust, alpha=1, beta=+0.20),
    "bri+30": partial(F.brightness_contrast_adjust, alpha=1, beta=+0.30),
    "con-30": partial(F.brightness_contrast_adjust, alpha=0.70, beta=0),
    "con-20": partial(F.brightness_contrast_adjust, alpha=0.80, beta=0),
    "con-10": partial(F.brightness_contrast_adjust, alpha=0.90, beta=0),
    "con+10": partial(F.brightness_contrast_adjust, alpha=1.10, beta=0),
    "con+20": partial(F.brightness_contrast_adjust, alpha=1.20, beta=0),
    "con+30": partial(F.brightness_contrast_adjust, alpha=1.30, beta=0),
    "exp-3": partial(F.gamma_transform, gamma=0.70),
    "exp-2": partial(F.gamma_transform, gamma=0.80),
    "exp-1": partial(F.gamma_transform, gamma=0.90),
    "exp+1": partial(F.gamma_transform, gamma=1.10),
    "exp+2": partial(F.gamma_transform, gamma=1.20),
    "exp+3": partial(F.gamma_transform, gamma=1.30),
    "hue-30": partial(F.shift_hsv, hue_shift=-30, sat_shift=0, val_shift=0),
    "hue-20": partial(F.shift_hsv, hue_shift=-20, sat_shift=0, val_shift=0),
    "hue-10": partial(F.shift_hsv, hue_shift=-10, sat_shift=0, val_shift=0),
    "hue+10": partial(F.shift_hsv, hue_shift=+10, sat_shift=0, val_shift=0),
    "hue+20": partial(F.shift_hsv, hue_shift=+20, sat_shift=0, val_shift=0),
    "hue+30": partial(F.shift_hsv, hue_shift=+30, sat_shift=0, val_shift=0),
    "sat-30": partial(F.shift_hsv, hue_shift=0, sat_shift=-30, val_shift=0),
    "sat-20": partial(F.shift_hsv, hue_shift=0, sat_shift=-20, val_shift=0),
    "sat-10": partial(F.shift_hsv, hue_shift=0, sat_shift=-10, val_shift=0),
    "sat+10": partial(F.shift_hsv, hue_shift=0, sat_shift=+10, val_shift=0),
    "sat+20": partial(F.shift_hsv, hue_shift=0, sat_shift=+20, val_shift=0),
    "sat+30": partial(F.shift_hsv, hue_shift=0, sat_shift=+30, val_shift=0),
    "val-30": partial(F.shift_hsv, hue_shift=0, sat_shift=0, val_shift=-30),
    "val-20": partial(F.shift_hsv, hue_shift=0, sat_shift=0, val_shift=-20),
    "val-10": partial(F.shift_hsv, hue_shift=0, sat_shift=0, val_shift=-10),
    "val+10": partial(F.shift_hsv, hue_shift=0, sat_shift=0, val_shift=+10),
    "val+20": partial(F.shift_hsv, hue_shift=0, sat_shift=0, val_shift=+20),
    "val+30": partial(F.shift_hsv, hue_shift=0, sat_shift=0, val_shift=+30),
}


def evaluation(dataset, method):
    average_mse_lab = 0
    average_ssim = 0
    average_mse_lab_gain = defaultdict(float)
    average_ssim_gain = defaultdict(float)

    for left_path, right_path in tqdm(dataset):
        left_gt = io.imread(left_path)
        right = io.imread(right_path)

        for distortion_name, distortion in distortions.items():
            left_distorted = distortion(left_gt)

            left_corrected = img_as_ubyte(method(img_as_float(left_distorted), img_as_float(right)))

            mse_lab_before = mse(rgb2lab(left_distorted), rgb2lab(left_gt))
            mse_lab_after = mse(rgb2lab(left_corrected), rgb2lab(left_gt))

            mse_lab_gain = mse_lab_before - mse_lab_after

            ssim_before = ssim(left_distorted, left_gt, channel_axis=-1)
            ssim_after = ssim(left_corrected, left_gt, channel_axis=-1)

            ssim_gain = ssim_after - ssim_before

            average_mse_lab += mse_lab_after
            average_ssim += ssim_after
            average_mse_lab_gain[distortion_name] += mse_lab_gain
            average_ssim_gain[distortion_name] += ssim_gain

    average_mse_lab /= len(dataset) * len(distortions)
    average_ssim /= len(dataset) * len(distortions)
    average_mse_lab_gain = {k: v / len(dataset) for k, v in average_mse_lab_gain.items()}
    average_ssim_gain = {k: v / len(dataset) for k, v in average_ssim_gain.items()}

    return average_mse_lab, average_ssim, average_mse_lab_gain, average_ssim_gain


if __name__ == "__main__":
    image_dir = Path("../datasets/dataset/Test")
    lefts = sorted(image_dir.glob("*_L.png"))
    rights = sorted(image_dir.glob("*_R.png"))

    assert len(lefts) == len(rights)

    dataset = list(zip(lefts, rights))

    average_mse_lab, average_ssim, average_mse_lab_gain, average_ssim_gain = evaluation(dataset, color_transfer_between_images)
    print(average_mse_lab, average_ssim)

    plt.title("Color Transfer Between Images")
    plt.bar(range(len(average_mse_lab_gain)), list(average_mse_lab_gain.values()), align='center')
    plt.ylabel("Average MSE LAB Gain")
    plt.xticks(range(len(average_mse_lab_gain)), list(average_mse_lab_gain.keys()), rotation=60)
    plt.show()

    plt.title("Color Transfer Between Images")
    plt.bar(range(len(average_ssim_gain)), list(average_ssim_gain.values()), align='center')
    plt.ylabel("Average SSIM Gain")
    plt.xticks(range(len(average_ssim_gain)), list(average_ssim_gain.keys()), rotation=60)
    plt.show()
