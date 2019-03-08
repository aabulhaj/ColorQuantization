import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray

GRAY_SCALE = 1
RGB = 2

YIQ_MAT = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
YIQ_MAT_INV = np.linalg.inv(YIQ_MAT)


def read_image(filename, representation):
    """Reads and normalizes the image.
    :param filename: Image path.
    :param representation: [grayscale=1, rgb=2]
    :return: Normalized image in the given representation.
    """
    normalized_image = imread(filename).astype(np.float64) / 255

    if representation == RGB:
        return normalized_image

    if representation == GRAY_SCALE:
        return rgb2gray(normalized_image)

    raise ValueError("Error: Representation should be 1 for gray scale, 2 for RGB.")


def imdisplay(image, representation):
    """Plots the given image."""
    plt.figure()
    if representation == GRAY_SCALE:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def rgb2yiq(imRGB):
    """Transforms a copy of the given RGB image into YIQ color space.
    :param imRGB: Height x width x 3 np.float64 matrix with values in [0, 1]. The red channel is encoded in
        imRGB[:, :, 0], the green in imRGB[:, :, 1], and the blue in imRGB[:, :, 2].
    :return: Image in YIQ color space (copy of the given image).
    """
    imYIQ = imRGB.copy()
    imYIQ[:, :, 0] = np.dot(imRGB, YIQ_MAT[0])
    imYIQ[:, :, 1] = np.dot(imRGB, YIQ_MAT[1])
    imYIQ[:, :, 2] = np.dot(imRGB, YIQ_MAT[2])

    imYIQ[imYIQ < 0] = 0
    imYIQ[imYIQ > 1] = 1
    return imYIQ


def yiq2rgb(imYIQ):
    """Transform a copy of the given image in the YIQ color space into an RGB image.
    :param imYIQ: Height x width x 3 np.float64 matrix with values in [0, 1]. imYIQ[:,:,0] encodes the luminance
        channel Y, imYIQ[:,:,1] encodes I, and imYIQ[:,:,2] encodes Q.
    :return: RGB image (copy of the given image).
    """
    imRGB = imYIQ.copy()
    imRGB[:, :, 0] = np.dot(imYIQ, YIQ_MAT_INV[0])
    imRGB[:, :, 1] = np.dot(imYIQ, YIQ_MAT_INV[1])
    imRGB[:, :, 2] = np.dot(imYIQ, YIQ_MAT_INV[2])

    imRGB[imRGB < 0] = 0
    imRGB[imRGB > 1] = 1
    return imRGB


def get_histogram(image):
    """Returns the histogram for a given image. If you need to calculate histogram for RGB image you should convert it
    to YIQ and return the image with Y values only. For example _get_hist_and_bins(YIQ_converted[:, 0]).
    :param image: The image can be RGB or YIQ.
    :return: Histogram for the given image.
    """
    return np.histogram(image, 256, [0, 1])[0]


def normalize_image(image):
    """Normalizes a given image in range between [0, 1] with float64 values.
    :param image: Image to be normalized
    :return: Normalized image between [0, 1] with float64 values.
    """
    return image.astype(np.float64) / 255
