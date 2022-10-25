import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image


def get_gaussian_kern(sigma=1.5, size=3):
    """
    size gotta be odd (e. g. 3, 5, 7, ...)
    p.s. could be rewritten with np.meshgrid()
    """
    indices = np.dstack(np.indices((size, size)))
    indices -= (size//2, size//2)  # offsetting coordinates

    x = indices[..., 1]
    y = indices[..., 0]

    normal = 1 / (2 * np.pi * sigma**2)
    gaussian_matrix = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / normal
    gaussian_matrix /= np.sum(gaussian_matrix)  # normalization helps to avoid oversaturation

    return gaussian_matrix

def rgb_convolve(image, kern):
    temp_image = np.empty_like(image)
    print("Color dim:", image.shape[-1])
    for dim in range(image.shape[-1]):
        temp_image = signal.convolve2d(image[:, :, dim],
                                       kern,
                                       mode="same",
                                       boundary="symm")
    return temp_image


plt.rcParams["figure.figsize"] = (10, 7)

img = plt.imread("image.jpg").astype(np.float) / 255.
print("Image loaded...")

kern = get_gaussian_kern()
print("Kernel is there...")

blurred_img = rgb_convolve(img, kern)
print("Filtered image...")

fig, (axL, axR) = plt.subplots(ncols=2, tight_layout=True)
fig.suptitle("Gaussian Blur")

axL.imshow(img)
axR.imshow(blurred_img)
plt.show()
