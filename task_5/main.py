import cv2
import numpy as np

img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (300, 300))


def add_noise(image, noise):
    im = image.copy()
    if im.shape != noise.shape:
        return None
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im[i, j] += noise[i, j]
            im[i, j] = np.clip(im[i, j], 0, 255)
    return im


def gen_noisy(img, N, type, *details):
    noisy = []
    row, col = img.shape
    for n in range(N):
        if type == 'gauss':
            gauss = np.random.normal(loc=details[0], scale=details[1], size=(row, col))
            noisy.append(add_noise(img, gauss))
        if type == 'uniform':
            uniform = np.random.uniform(low=details[0], high=details[1], size=(row, col))
            noisy.append(add_noise(img, uniform))
    return noisy


def average(im_arr):
    res = im_arr[0].copy()
    shape = im_arr[0].shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = 0
            for n in range(len(im_arr)):
                val += im_arr[n][i, j]
            res[i, j] = val // N
    return res


# number of noised images
N = 16

noisy = gen_noisy(img, N, 'gauss', 0.0, 16.0)
# noisy = gen_noisy(img, N, 'uniform', -64, 64)

av = average(noisy)

cv2.imshow('lenna', img)
cv2.imshow('noisy', noisy[0])
cv2.imshow('average', av)

cv2.waitKey(0)
