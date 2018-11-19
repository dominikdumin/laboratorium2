import numpy as np
import cv2 as cv
from skimage import morphology
from obpng import read_png, write_png
from scipy.ndimage import imread
import itertools
from read_image import read_image


# Zadanie na ocenę dostateczną
def renew_pictures():
    image = read_image('figures/crushed.png')

    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    renewed = morphology.erosion(image, selem=kernel)
    write_png(renewed, 'results/renewed.png')

    # ---

    image = read_image('figures/crushed2.png')
    renewed = morphology.erosion(image, selem=kernel)

    kernel = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ])
    renewed = morphology.erosion(image, selem=kernel)
    write_png(renewed, 'results/renewed2.png')


# Zadanie na ocenę dobrą
def own_simple_erosion(image):
    new_image = np.zeros(image.shape, dtype=image.dtype)

    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    y_steps = (kernel.shape[0] - 1) // 2
    x_steps = (kernel.shape[1] - 1) // 2

    for (y, x) in np.ndindex(image.shape):
        flag = False
        for ky in range(kernel.shape[0] - 1):
            for kx in range(kernel.shape[1] - 1):
                try:
                    val = image[y-ky+y_steps][x-kx+x_steps]
                    if val == 0:
                        flag = True
                except:
                    pass
        
        if flag:
            new_image[y][x] = 0
        else:
            new_image[y][x] = image[y][x]

    return new_image


# Zadanie na ocenę bardzo dobrą
def own_erosion(image, kernel=None):
    new_image = np.zeros(image.shape, dtype=image.dtype)
    if kernel is None:
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])

    y_steps = (kernel.shape[0] - 1) // 2
    x_steps = (kernel.shape[1] - 1) // 2

    for (y, x) in np.ndindex(image.shape):
        flag = False
        for ky in range(kernel.shape[0] - 1):
            for kx in range(kernel.shape[1] - 1):
                try:
                    val = image[y-ky+y_steps][x-kx+x_steps]
                    if val == 0:
                        flag = True
                except:
                    pass
        
        if flag:
            new_image[y][x] = 0
        else:
            new_image[y][x] = image[y][x]

    return new_image
