import numpy as np
import matplotlib.pyplot as plt
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


def gaussian_blur(image: Image, sigma=1.5, size=3):
    a = np.array(image)
    kern = get_gaussian_kern(sigma, size)

    a = np.pad(a, ((size//2, size//2), (size//2, size//2), (0, 0)), mode="edge")

    height, width = a.shape[:2]
    for y in range(size//2, height-size//2):
        for x in range(size//2, width-size//2):
            area = a[y-size//2:y+size//2+1, x-size//2:x+size//2+1]

            red = np.ceil(np.sum(area[..., 0] * kern)).astype("uint8")
            green = np.ceil(np.sum(area[..., 1] * kern)).astype("uint8")
            blue = np.ceil(np.sum(area[..., 2] * kern)).astype("uint8")

            a[y, x] = np.dstack((red, green, blue))
    return a[size//2:-size//2, size//2:-size//2]


img = Image.open("image.jpg")
blurred_img = gaussian_blur(img, sigma=20, size=11)

fig, axs = plt.subplots(1, 2)

axs[0].set_title("Original")
axs[0].imshow(img)
axs[0].axis("off")

axs[1].set_title("Gaussian Blur")
axs[1].imshow(blurred_img)
axs[1].axis("off")

plt.show()
