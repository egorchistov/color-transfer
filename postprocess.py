"""Real-World Stereo Color and Sharpness Mismatch Dataset Postprocessing Script

This script is used for our dataset postprocessing as described
at https://videoprocessing.ai/datasets/stereo-mismatch.html#methodology

Download dataset sources, adjust json configuration files if you need,
and then run this script.
"""

import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import SIFT, match_descriptors
from skimage.util import img_as_float, img_as_ubyte
from kornia.feature import LoFTR
from kornia.utils import image_to_tensor

from methods.linear import monge_kantorovitch_color_transfer as mkct


def parse_args():
    parser = argparse.ArgumentParser(description="Script for processing all samples")
    parser.add_argument("--root", type=str, required=True, help="Path to folder with all samples")
    parser.add_argument("--samples", type=str, help="Samples to process", required=False)
    parser.add_argument("--output", type=str, required=True, help="Path to output folder to save processed samples")
    parser.add_argument("--frames", type=int, default=50, help="How many frames select from each sample")

    return parser.parse_args()


def extract(image):
    extractor = SIFT(upsampling=1)
    extractor.detect_and_extract(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    extractor.keypoints = extractor.keypoints[:, ::-1]

    return extractor


def estimate_homography(target, reference, method="SIFT"):
    if method == "SIFT":
        reference = extract(reference)
        target = extract(target)    

        matches = match_descriptors(target.descriptors, reference.descriptors)

        target_keypoints = target.keypoints[matches[:, 0]]
        reference_keypoints = reference.keypoints[matches[:, 1]]
    elif method == "LOFTR":
        scale = np.array([target.shape[1] / 512, target.shape[0] / 512])

        target = cv2.resize(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), (512, 512))
        reference = cv2.resize(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (512, 512))

        matcher = LoFTR(pretrained="indoor")
        correspondences_dict = matcher({
            "image0": image_to_tensor(img_as_float(target), keepdim=False).float(),
            "image1": image_to_tensor(img_as_float(reference), keepdim=False).float()
        })

        target_keypoints = correspondences_dict["keypoints0"].numpy() * scale
        reference_keypoints = correspondences_dict["keypoints1"].numpy() * scale
    else:
        raise ValueError(f"Unknown method: {method}")

    homography, _ = cv2.findHomography(
        target_keypoints,
        reference_keypoints,
        method=cv2.USAC_MAGSAC)

    return homography


def frames(args, params, sample):
    cap_left = cv2.VideoCapture(os.path.join(args.root, sample, "left.mp4"))
    cap_left_gt = cv2.VideoCapture(os.path.join(args.root, sample, "left_gt.mp4"))
    cap_right = cv2.VideoCapture(os.path.join(args.root, sample, "right.mp4"))

    if not all(cap.isOpened() for cap in (cap_left, cap_left_gt, cap_right)):
        print("Can not open source files for sample:", sample)
        return

    cap_left.set(cv2.CAP_PROP_POS_FRAMES, params["offsets"]["all"] + params["offsets"]["left"])
    cap_left_gt.set(cv2.CAP_PROP_POS_FRAMES, params["offsets"]["all"] + params["offsets"]["left_gt"])
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, params["offsets"]["all"] + params["offsets"]["right"])

    for frame_idx in range(args.frames):
        _, left = cap_left.read()
        _, left_gt = cap_left_gt.read()
        _, right = cap_right.read()

        # Horizontally flip frame because mirror flips
        left = cv2.flip(left, flipCode=1)

        yield frame_idx, left, left_gt, right

    cap_left.release()
    cap_left_gt.release()
    cap_right.release()


if __name__ == "__main__":
    args = parse_args()

    if isinstance(args.samples, str):
        args.samples = args.samples.split(",")
    else:
        args.samples = sorted(os.listdir(args.root))

    for sample in tqdm(args.samples, desc="Samples"):
        with open(os.path.join(args.root, sample, "params.json"), "r") as f:
            params = json.load(f)

        bbox = params["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        for frame_idx, left, left_gt, right in frames(args, params, sample):
            if frame_idx == 0:
                H1 = estimate_homography(left, left_gt)
                H2 = estimate_homography(right, left_gt, method="LOFTR")

                left_gt_crop = left_gt[y: y + h, x: x + w]
                right_crop = cv2.warpPerspective(right, H2, (right.shape[1], right.shape[0]))
                right_crop = right_crop[y: y + h, x: x + w]

                H3 = estimate_homography(right_crop, left_gt_crop, method="LOFTR")
                H3 = np.array([[1, 0, H3[0, 2]], [0, 1, 0], [0, 0, 1]])

                H2 = H2 @ H3

            left = cv2.warpPerspective(left, H1, (left.shape[1], left.shape[0]))
            right = cv2.warpPerspective(right, H2, (right.shape[1], right.shape[0]))

            left = left[y: y + h, x: x + w]
            left_gt = left_gt[y: y + h, x: x + w]
            right = right[y: y + h, x: x + w]

            right = img_as_ubyte(mkct(img_as_float(right), img_as_float(left_gt)).clip(0, 1))

            os.makedirs(os.path.join(args.output, sample), exist_ok=True)

            cv2.imwrite(os.path.join(args.output, sample, f"{frame_idx:04}_LD.png"), left)
            cv2.imwrite(os.path.join(args.output, sample, f"{frame_idx:04}_L.png"), left_gt)
            cv2.imwrite(os.path.join(args.output, sample, f"{frame_idx:04}_R.png"), right)
