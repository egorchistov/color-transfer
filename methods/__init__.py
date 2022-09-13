"""Colot Transfer Methods

Color transfer is the method of transforming the color of a target image
so that the color becomes more consistent with the reference image.

All methods implement the same interface with target and reference arguments:

corrected = method(target, reference)

corrected image will preserve structure from target image and color from reference.

"""

from pathlib import Path

import cv2
from skimage import img_as_float, img_as_ubyte
from tqdm import tqdm


def runner(target_mask, reference_mask, corrected_mask, method):
    """Color Transfer Runner

    This runner can be used to apply any color transfer method to a video
    or a frame sequence. It used OpenCV backend and accept many formats.

    >>> from methods.linear import color_transfer_in_correlated_color_space as ct_ccs
    >>> runner("target.png", "reference.png", "corrected.png", ct_ccs)
    >>> runner("target/%04d.png", "reference/%04d.png", "corrected/%04d.png", ct_ccs)
    >>> runner("target.mp4", "reference.mp4", "corrected.mp4", ct_ccs)
    """
    Path(corrected_mask).parent.mkdir(parents=True, exist_ok=True)

    cap_target = cv2.VideoCapture(target_mask)
    cap_reference = cv2.VideoCapture(reference_mask)

    width = int(cap_target.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_target.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap_target.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap_target.get(cv2.CAP_PROP_FPS)) if fourcc else 0
    cap_corrected = cv2.VideoWriter(corrected_mask, fourcc, fps, (width, height))

    frame_count = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        with tqdm(desc="Frames", total=frame_count) as pbar:
            while all(cap.isOpened() for cap in (cap_target, cap_reference, cap_corrected)):
                ret_target, target = cap_target.read()
                ret_reference, reference = cap_reference.read()

                if not ret_target or not ret_reference:
                    break

                corrected = img_as_ubyte(method(img_as_float(target), img_as_float(reference)).clip(0, 1))

                cap_corrected.write(corrected)
                pbar.update(1)
    finally:
        cap_target.release()
        cap_reference.release()
        cap_corrected.release()
