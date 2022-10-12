import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_gaussian_kern(sigma=1.5, size=3):
    """
    size gotta be odd (3, 5, 7, ...)
    """
    indices = np.dstack(np.indices((size, size)))
    indices -= (size//2, size//2)  # offsetting coordinates

    x = indices[..., 1]
    y = indices[..., 0]

    gaussian_matrix = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (np.pi * 2 * sigma**2)
    gaussian_matrix /= np.sum(gaussian_matrix)  # normalization helps to avoid oversaturation

    return gaussian_matrix


def gaussian_blur(pixels, sigma=1.5, size=3):
    pixels = np.asarray(pixels)
    kern = get_gaussian_kern(sigma, size)

    height, width = pixels.shape[:2]
    for x in range(1, width-1):
        for y in range(1, height-1):
            center = y+1, x+1

            area = pixels[y-1:y+2, x-1:x+2]

            red = np.ceil(np.sum(area[..., 0] * kern)).astype("uint8")
            green = np.ceil(np.sum(area[..., 1] * kern)).astype("uint8")
            blue = np.ceil(np.sum(area[..., 2] * kern)).astype("uint8")

            pixels[center] = np.dstack((red, green, blue))
    return pixels


image = Image.open("image.jpg")
pixels = np.array(image)
pixels = gaussian_blur(pixels)
plt.imshow(pixels)
plt.show()
