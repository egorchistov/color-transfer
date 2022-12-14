# Color Transfer

Color transfer is a&nbsp;method of&nbsp;transforming the color of&nbsp;a&nbsp;target image so that it becomes consistent with the color of&nbsp;the&nbsp;reference image. This repo is&nbsp;an&nbsp;open source color transfer methods and datasets collection. Feel free to&nbsp;submit any method or&nbsp;dataset.

## Key Features

* 5 available methods with common interface
* 2 available datasets with the same structure
* Easy to use interface to apply methods to images or videos

## Available Methods

1. Reinhard&nbsp;et&nbsp;al., "Color Transfer between Images", 2001

    ```python
    from methods.linear import color_transfer_between_images as ct
    ```

2. Xiao&nbsp;et&nbsp;al., "Color Transfer in&nbsp;Correlated Color Space", 2006

    ```python
    from methods.linear import color_transfer_in_correlated_color_space as ct_ccs
    ```

3. Pitie&nbsp;et&nbsp;al., "The Linear Monge-Kantorovitch Linear Colour Mapping for&nbsp;Example-Based Colour Transfer", 2007

    ```python
    from methods.linear import monge_kantorovitch_color_transfer as mkct
    ```

4. Pitie&nbsp;et&nbsp;al., "Automated Colour Grading using Colour Distribution Transfer", 2007

    ```python
    from methods.iterative import automated_color_grading as acg
    ```

5. Croci&nbsp;et&nbsp;al., "Deep Color Mismatch Correction in&nbsp;Stereoscopic 3D Images", 2021

    ```python
    from methods.dcmc import deep_color_mismatch_correction as dcmc
    ```

## How to Use

First clone this repo and install dependencies:

```shell
git clone git@github.com:egorchistov/color-transfer.git
cd color-transfer
pip install -r requirements.txt
```

Then use this simple python script to&nbsp;run color transfer:

```python
from methods import runner
# You can use other methods instead. See available methods above
from methods.linear import monge_kantorovitch_color_transfer as mkct

# You can use this runner to apply any color transfer method to a video or a frame sequence.
# It uses OpenCV backend and accepts many formats.
runner("images/0_target.png", "images/0_reference.png", "images/0_mkct.png", mkct)
# runner("target/%04d.png", "reference/%04d.png", "corrected/%04d.png", mkct)
# runner("target.mp4", "reference.mp4", "corrected.mp4", mkct)
```

## Available Datasets

Below you can find two processed datasets. Preparation scripts can be found in datasets folder

1. [Stereo Color and Sharpness Mismatch Dataset](https://videoprocessing.ai/datasets/stereo-mismatch.html)
2. [Deep Color Mismatch Correction in Stereoscopic 3D Images Dataset](https://www.kaggle.com/datasets/egorchistov/dcmc-dataset)

