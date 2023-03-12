"""Colot Mismatch Correction Methods

Color-mismatch correction is the task of transferring color from one view
of a stereopair to corresponding areas in another where the colors differ
incorrectly.

All methods implement the same interface with target and reference arguments:

corrected = method(target, reference)

The corrected view should have a structure consistent with that of the target view and
colors consistent with those of the reference view.
"""

from pathlib import Path

import cv2
import torch
from skimage.util import img_as_float, img_as_ubyte
from kornia import image_to_tensor, tensor_to_image
from tqdm import tqdm


@torch.no_grad()
def run_nn(target, reference, device, model):
    target = image_to_tensor(target, keepdim=False).float().to(device)
    reference = image_to_tensor(reference, keepdim=False).float().to(device)

    corrected_left, _ = model(target, reference)

    return tensor_to_image(corrected_left)


def runner(target_mask, reference_mask, corrected_mask, method):
    """This runner can be used to apply any method to a video
    or a frame sequence. It used OpenCV backend and accept many formats.

    >>> from methods.linear import monge_kantorovitch_color_transfer as mkct
    >>> runner("graphics/%04d_LD.png", "graphics/%04d_R.png", "graphics/%04d_MKCT.png", mkct)
    >>> runner("LD.mp4", "R.mp4", "MKCT.mp4", mkct)
    """
    Path(corrected_mask).parent.mkdir(parents=True, exist_ok=True)

    cap_target = cv2.VideoCapture(str(target_mask))
    cap_reference = cv2.VideoCapture(str(reference_mask))

    width = int(cap_target.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_target.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap_target.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap_target.get(cv2.CAP_PROP_FPS)) if fourcc else 0
    cap_corrected = cv2.VideoWriter(str(corrected_mask), fourcc, fps, (width, height))

    frame_count = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        with tqdm(desc="Frames", total=frame_count) as pbar:
            while all(cap.isOpened() for cap in (cap_target, cap_reference, cap_corrected)):
                ret_target, target = cap_target.read()
                ret_reference, reference = cap_reference.read()

                if not ret_target or not ret_reference:
                    break

                target = img_as_float(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
                reference = img_as_float(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
                corrected = method(target, reference).clip(0, 1)
                corrected = cv2.cvtColor(img_as_ubyte(corrected), cv2.COLOR_RGB2BGR)

                cap_corrected.write(corrected)
                pbar.update(1)
    finally:
        cap_target.release()
        cap_reference.release()
        cap_corrected.release()
