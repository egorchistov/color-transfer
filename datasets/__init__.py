"""Dataset Preparation

Download datasets from

* [Flickr1024](https://yingqianwang.github.io/Flickr1024/)
* [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
* [stereoscopic_images](http://ivylabprev.kaist.ac.kr/demo/3DVCA/3DVCA.htm)

and place in this folder like this

Flick1024/Test/
Flick1024/Train/
Flick1024/Validation/
InStereo2K/test/
InStereo2K/train/
stereoscopic_images/

Then you can use provided color_distortion.csv to select used image pairs.

Metric values can be interpreted as:

* 2.00 — Scenes with virtually no color mismatch
* 6.00 — Scenes with some color mismatches

Confidence is measured in interval 0-255, where 255 is a maximum confidence.

We select 0.9 * 255 confidence threshold with color distortion < 2.08 to get 1035 pairs.

MSU 3DColor 2017 metric is a part of [VQMT3D Project](https://videoprocessing.ai/stereo_quality/).
For MSU 3DColor 2017 metric calculation you need VQMT3D binary.

80% of the dataset was selected for the training set, 10% for the validation set,
and another 10% for the test set:

dataset/Test
dataset/Train
dataset/Validation

We do not shuffle data before split because InStereo2K contains sequential frames.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def color_distortion(left_path, right_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.run([
            str(Path("VQMT3D/VQMT3D")),
            "--metrics=color_distortion",
            "--output=" + tmpdirname,
            str(left_path),
            str(right_path)
        ],
            check=True,
            capture_output=True)

        tmpdirname = Path(tmpdirname)

        with open(tmpdirname / "color_distortion.txt") as file:
            _, value = file.readline().split()

            value = float(value)

        with open(tmpdirname / "color_distortion_confidence.txt") as file:
            _, confidence = file.readline().split()

            confidence = float(confidence)

        return value, confidence


def calculate_dataset_color_distortion(dataset):
    records = []

    for left_path, right_path in tqdm(dataset):
        value, confidence = color_distortion(left_path, right_path)
        records.append({
            "left_path": left_path,
            "right_path": right_path,
            "MSU 3DColor 2017": value,
            "MSU 3DColor 2017 Confidence": confidence
        })

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    flick1024_dir = Path("Flickr1024")
    instereo2k_dir = Path("InStereo2K")
    stereoscopic_images_dir = Path("stereoscopic_images")
    output = Path("dataset")

    if not Path("color_distortion.csv").exists():
        lefts = []
        rights = []

        lefts.extend(sorted(flick1024_dir.glob("*/*_L.png")))
        rights.extend(sorted(flick1024_dir.glob("*/*_R.png")))

        lefts.extend(sorted(instereo2k_dir.glob("*/*/left.png")))
        rights.extend(sorted(instereo2k_dir.glob("*/*/right.png")))

        lefts.extend(sorted(stereoscopic_images_dir.glob("*_[lL]*")))
        rights.extend(sorted(stereoscopic_images_dir.glob("*_[rR]*")))

        assert len(lefts) == len(rights)

        color_distortion = calculate_dataset_color_distortion(list(zip(lefts, rights)))

        color_distortion["Use"] = (color_distortion["MSU 3DColor 2017 Confidence"] > 255 * 0.9) & \
                                  (color_distortion["MSU 3DColor 2017"] < 2.08)

        color_distortion.to_csv("color_distortion.csv", index=False)
    else:
        color_distortion = pd.read_csv("color_distortion.csv")

    output.mkdir(exist_ok=True)

    for index, (_, row) in tqdm(enumerate(color_distortion[color_distortion.Use].iterrows())):
        shutil.copy2(row.left_path, output / (f"{index + 1:04}_L" + Path(row.left_path).suffix))
        shutil.copy2(row.right_path, output / (f"{index + 1:04}_R" + Path(row.right_path).suffix))
