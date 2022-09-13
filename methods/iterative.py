"""Iterative Color Transfer Methods"""

import scipy
import numpy as np
from skimage.transform import resize


def iterative_distribution_transfer(target, reference, bins=255, n_iter=4):
    """Iterative Disribution Transfer

    This method iteratively project target and reference images on random 1D axes
    and perform probability density transfer according to that axis.

    Links
    -----
    https://github.com/ptallada/colour_transfer

    Citation
    --------
    @article{pitie2007automated,
      title={Automated colour grading using colour distribution transfer},
      author={Piti{\'e}, Fran{\c{c}}ois and Kokaram, Anil C and Dahyot, Rozenn},
      journal={Computer Vision and Image Understanding},
      volume={107},
      number={1-2},
      pages={123--137},
      year={2007},
      publisher={Elsevier}
    }
    """
    shape = target.shape
    n_dims = shape[-1]

    target = target.reshape(-1, 3)
    reference = reference.reshape(-1, 3)

    for _ in range(n_iter):
        r = scipy.stats.special_ortho_group.rvs(n_dims)

        d0r = r @ target.T
        d1r = r @ reference.T
        d_r = np.empty_like(target.T)

        for j in range(n_dims):
            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(float)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(float)
            cp1r /= cp1r[-1]

            f = np.interp(cp0r, cp1r, edges[1:])

            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        target = np.linalg.solve(r, (d_r - d0r)).T + target

    output = target.reshape(shape)

    return output


def _regrain(img_arr_in, img_arr_col, nbits=(4, 16, 32, 64, 64, 64), level=0):
    h, w, _ = img_arr_in.shape
    h2 = (h + 1) // 2
    w2 = (w + 1) // 2

    if len(nbits) > 1 and h2 > 20 and w2 > 20:
        resize_arr_in = resize(img_arr_in, (h2, w2))
        resize_arr_col = resize(img_arr_col, (h2, w2))
        resize_arr_out = _regrain(resize_arr_in, resize_arr_col, nbits[1:], level + 1)
        img_arr_out = resize(resize_arr_out, (h, w))
    else:
        img_arr_out = img_arr_in

    img_arr_out = _solve(img_arr_out, img_arr_in, img_arr_col, nbits[0], level)

    return img_arr_out


def _solve(img_arr_out,
           img_arr_in,
           img_arr_col,
           nbit,
           level,
           eps=1e-6):
    width, height, c = img_arr_in.shape
    first_pad_0 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)
    first_pad_1 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
    last_pad_0 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
    last_pad_1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)

    delta_x = last_pad_1(img_arr_in) - first_pad_1(img_arr_in)
    delta_y = last_pad_0(img_arr_in) - first_pad_0(img_arr_in)
    delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

    psi = 256 * delta / 5
    psi[psi > 1] = 1
    phi = 30 * 2**(-level) / (1 + 10 * delta)

    phi1 = (last_pad_1(phi) + phi) / 2
    phi2 = (last_pad_0(phi) + phi) / 2
    phi3 = (first_pad_1(phi) + phi) / 2
    phi4 = (first_pad_0(phi) + phi) / 2

    rho = 1 / 5.0
    for i in range(nbit):
        den = psi + phi1 + phi2 + phi3 + phi4
        num = (np.tile(psi, [1, 1, c]) * img_arr_col +
               np.tile(phi1, [1, 1, c]) * (last_pad_1(img_arr_out) - last_pad_1(img_arr_in) + img_arr_in) +
               np.tile(phi2, [1, 1, c]) * (last_pad_0(img_arr_out) - last_pad_0(img_arr_in) + img_arr_in) +
               np.tile(phi3, [1, 1, c]) * (first_pad_1(img_arr_out) - first_pad_1(img_arr_in) + img_arr_in) +
               np.tile(phi4, [1, 1, c]) * (first_pad_0(img_arr_out) - first_pad_0(img_arr_in) + img_arr_in))
        img_arr_out = (num / np.tile(den + eps, [1, 1, c]) * (1 - rho) + rho * img_arr_out)

    return img_arr_out


def automated_color_grading(target, reference):
    """Automated Colour Grading using Colour Distribution Transfer

    This method iteratively project target and reference images on random 1D axes
    and perform probability density transfer according to that axis. Then authors
    reduce grain noise artifacts by minimizing cost function.

    Links
    -----
    https://github.com/pengbo-learn/python-color-transfer

    Citation
    --------
    @article{pitie2007automated,
      title={Automated colour grading using colour distribution transfer},
      author={Piti{\'e}, Fran{\c{c}}ois and Kokaram, Anil C and Dahyot, Rozenn},
      journal={Computer Vision and Image Understanding},
      volume={107},
      number={1-2},
      pages={123--137},
      year={2007},
      publisher={Elsevier}
    }
    """
    output = iterative_distribution_transfer(target, reference)
    output = _regrain(target, output)

    return output
