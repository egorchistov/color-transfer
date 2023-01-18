"""Augment test dataset

We use undistored image pairs and then introduce color mismatch,
applying different color modification operators to one of the images.

Then we crop the center of the image using

```shell
mogrify -gravity center -crop 512x512+0+0 +repage *.png
```

"""

from pathlib import Path

from skimage import io
from tqdm import tqdm

from data import distortions


if __name__ == "__main__":
    image_dir = Path("dataset/Test")
    lefts = sorted(image_dir.glob("*_L.png"))

    for left_path in tqdm(lefts):
        left_gt = io.imread(left_path)
        left_distorted = distortions(image=left_gt)["image"]

        fname = str(left_path).replace("L", "LD")
        io.imsave(fname, left_distorted)
