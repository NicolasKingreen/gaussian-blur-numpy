import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image = Image.open("image.jpg")
pixels = np.array(image)
size = height, width = pixels.shape[:2]


for x in range(1, width-1):
    for y in range(1, height-1):
        center = y+1, x+1

        # take a square
        area = pixels[y-1:y+2, x-1:x+2]

        # apply gaussian window
        gaussian_matrix_indices = np.dstack(np.indices(area.shape[:2]))
        gaussian_matrix_indices -= (1, 1)

        gaussian_matrix_x = gaussian_matrix_indices[..., 1]
        gaussian_matrix_y = gaussian_matrix_indices[..., 0]

        SIGMA = 1
        gaussian_matrix = np.exp(-(gaussian_matrix_x**2 + gaussian_matrix_y**2) / (2 * SIGMA**2)) / (np.pi * 2 * SIGMA**2)
        gaussian_matrix /= np.sum(gaussian_matrix)

        red_blur = np.ceil(np.sum(area[..., 0] * gaussian_matrix)).astype("uint8")
        green_blur = np.ceil(np.sum(area[..., 1] * gaussian_matrix)).astype("uint8")
        blue_blur = np.ceil(np.sum(area[..., 2] * gaussian_matrix)).astype("uint8")

        pixels[center] = np.dstack((red_blur, green_blur, blue_blur))

plt.imshow(pixels)
plt.show()
