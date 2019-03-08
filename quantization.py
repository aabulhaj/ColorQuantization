import numpy as np

import utils


def _calculate_initial_z(histogram, n_quant):
    # Each segment should have approximately the same number of pixels.
    cum_hist = np.cumsum(histogram).astype(np.float64)

    cum_hist = np.rint((cum_hist / np.max(cum_hist)) * (n_quant - 1))
    segments = np.flatnonzero(np.r_[1, np.diff(cum_hist)[:-1]])

    return np.append(segments, [255]).astype(np.float64)


def _get_z(q, z):
    for i in range(1, len(q)):
        z[i] = (q[i - 1] + q[i]) / 2
    return z


def _get_q(z, q, histogram):
    for i in range(len(z) - 1):
        lower_bound = np.round(z[i]).astype(np.uint8)
        upper_bound = np.round(z[i + 1]).astype(np.uint8)
        q[i] = np.sum(histogram[lower_bound:upper_bound] * np.arange(lower_bound, upper_bound)) / np.sum(
            histogram[lower_bound:upper_bound])
    return q


def quantize(im_orig, n_quant, n_iter):
    """Performs optimal quantization of a grayscale or RGB image.
    :param im_orig: Input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: Number of intensities the output im_quant image should have.
    :param n_iter: Maximum number of iterations of the optimization procedure (may converge earlier).
    :return: Quantized output image (copy of the original image).
    """
    if n_quant <= 0 or n_iter <= 0:
        raise ValueError("Error: n_quant and n_iter must be positive")

    if im_orig.ndim == 3:
        img = utils.rgb2yiq(im_orig)
        img_hist = utils.get_histogram(img[:, :, 0])
    else:
        img = im_orig.copy()
        img_hist = utils.get_histogram(img)

    q = np.zeros(n_quant).astype(np.float64)
    z = _calculate_initial_z(img_hist, n_quant)
    last_iter_z = z.copy()
    for i in range(n_iter):
        q = _get_q(z, q, img_hist)
        z = _get_z(q, z)

        # Checks for convergence.
        if np.array_equal(last_iter_z, z):
            break
        last_iter_z = z.copy()

    lookup_table = np.zeros(256)
    for i in range(len(z) - 1):
        start = int(np.round(z[i]))
        end = int(z[i + 1]) + 1
        lookup_table[start:end] = q[i]

    if im_orig.ndim == 3:
        img[:, :, 0] = utils.normalize_image(lookup_table[np.rint(img[:, :, 0] * 255).astype(np.uint8)])
        return utils.yiq2rgb(img)
    return utils.normalize_image(lookup_table[np.rint(img * 255).astype(np.uint8)])

# Run example:

# REPR = utils.GRAY_SCALE
# N_QUANTS = 3
# ITERATIONS = 1000
#
# image = utils.read_image('images/gray_orig.png', REPR)
# im = quantize(image, N_QUANTS, ITERATIONS)
# utils.imdisplay(im, REPR)
