"""Linear Color Transfer Methods

The linear color transfer methods approximate image as an three-dimensional Gaussian
distributed signal.
"""

import scipy
import numpy as np
from skimage.color import rgb2lab, lab2rgb


def color_transfer_between_images(target, reference):
    """Color Transfer between Images

    This method use the simple statistical representation of an image as normal
    distributed three-dimensional signal. Transfer from reference to target
    is straightforward:

    >>> corrected = (target - target_mean) * reference_std / target_std + reference_mean

    Because RGB color space is higly correlated, authors used LAB color space. Authors
    suggest to use clusterization algorithms in order to transfer color between same objects.

    Citation
    --------
    @article{reinhard2001color,
      title={Color transfer between images},
      author={Reinhard, Erik and Adhikhmin, Michael and Gooch, Bruce and Shirley, Peter},
      journal={IEEE Computer graphics and applications},
      volume={21},
      number={5},
      pages={34--41},
      year={2001},
      publisher={IEEE}
    }
    """
    target = rgb2lab(target)
    reference = rgb2lab(reference)

    shape = target.shape

    target = target.reshape(-1, 3)
    reference = reference.reshape(-1, 3)

    target_mean = np.mean(target, axis=0)
    reference_mean = np.mean(reference, axis=0)
    target_std = np.std(target, axis=0)
    reference_std = np.std(reference, axis=0)

    output = (target - target_mean) * reference_std / target_std + reference_mean

    output = lab2rgb(output.reshape(shape))

    return output


def color_transfer_in_correlated_color_space(target, reference):
    """Color Transfer in Correlated Color Space

    This method perform color transfer in correlated color space. To achieve this authors
    compute covariance between three color components, because covariane matrix is the
    extension of standard deviation in correlated space.

    Authors describe transformation in form of a scale, rotation and shift of pixel
    clusters. They decompose covariance matrix using SVD algorithm and determine
    transformation using geometric meaning of SVD.

    See also
    --------
    About geometric meaning of SVD
        https://en.wikipedia.org/wiki/Singular_value_decomposition

    Citation
    --------
    @inproceedings{xiao2006color,
      title={Color transfer in correlated color space},
      author={Xiao, Xuezhong and Ma, Lizhuang},
      booktitle={Proceedings of the 2006 ACM international conference on Virtual reality continuum and its applications},
      pages={305--309},
      year={2006}
    }
    """

    shape = target.shape

    target = target.reshape(-1, 3)
    reference = reference.reshape(-1, 3)

    target_mean = np.mean(target, axis=0)
    reference_mean = np.mean(reference, axis=0)
    target_cov = np.cov(target.T)
    reference_cov = np.cov(reference.T)

    target_u, target_s, _ = np.linalg.svd(target_cov)
    reference_u, reference_s, _ = np.linalg.svd(reference_cov)

    target_rotation = target_u
    reference_rotation = np.linalg.inv(reference_u)

    target_scale = np.diag(1 / np.sqrt(target_s))
    reference_scale = np.diag(np.sqrt(reference_s))

    T = target_rotation @ target_scale @ reference_scale @ reference_rotation

    output = (target - target_mean) @ T.T + reference_mean

    return output.reshape(shape)


def monge_kantorovitch_color_transfer(target, reference, decomposition="MK"):
    """The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer

    Authors generalize all linear color transfer approaches to this system of equations:

    corrected == T @ (target - target_mean) + reference_mean
    T @ cov_target T.T == cov_reference
    T = B @ Q @ A^-1 with Q.T @ Q = I

    where T can be decomposed in three different ways using:
    - "cholesky" — cholesky decomposition
    - "sqrt" — square root decomposition
    - "MK" — linear Monge-Kantorovitch solution

    Implementation has `decomposition` parameter to choose one of these approaches.

    Citation
    --------
    @article{pitie2007linear,
      title={The linear monge-kantorovitch linear colour mapping for example-based colour transfer},
      author={Piti{\'e}, Fran{\c{c}}ois and Kokaram, Anil},
      year={2007},
      publisher={IET}
    }
    """
    shape = target.shape

    target = target.reshape(-1, 3)
    reference = reference.reshape(-1, 3)

    target_mean = np.mean(target, axis=0)
    reference_mean = np.mean(reference, axis=0)
    target_cov = np.cov(target.T)
    reference_cov = np.cov(reference.T)

    if decomposition == "cholesky":
        A = np.linalg.cholesky(target_cov)
        B = np.linalg.cholesky(reference_cov)
        T = B @ np.linalg.inv(A)
    elif decomposition == "sqrt":
        A = scipy.linalg.sqrtm(target_cov)
        B = scipy.linalg.sqrtm(reference_cov)
        T = B @ np.linalg.inv(A)
    elif decomposition == "MK":
        A = scipy.linalg.sqrtm(target_cov)
        T = np.linalg.inv(A) @ scipy.linalg.sqrtm(A @ reference_cov @ A) @ np.linalg.inv(A)
    else:
        raise ValueError("Unknown decomposition")

    output = (target - target_mean) @ T + reference_mean

    return output.reshape(shape)
