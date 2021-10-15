from matplotlib import pyplot as plt
import numpy as np

import cv2

window_name = 'Alpha_Beta_Gamma'

gamma = 1.0
alpha = 1.0
beta = 0.0


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def update():
    global gamma, alpha, beta
    res = adjust_gamma(img, gamma)
    res = cv2.convertScaleAbs(res, alpha=alpha, beta=beta)
    cv2.imshow(window_name, res)


def set_gamma(pos):
    global img, gamma
    gamma = (pos + 1) / 100
    update()


def set_alpha(pos):
    global img, alpha
    alpha = (pos + 1) / 100
    update()


def set_beta(pos):
    global img, beta
    beta = pos
    update()


def nothing(x):
    pass


abg_flag = False
hist_flag = False
eq_flag = True

if abg_flag:
    img = cv2.imread('Lenna.png')
    img = cv2.resize(img, (400, 400))

    param_window_name = window_name
    cv2.namedWindow(param_window_name)
    cv2.createTrackbar('alpha', param_window_name, 10, 100, set_alpha)
    cv2.createTrackbar('beta', param_window_name, 100, 300, set_beta)
    cv2.createTrackbar('gamma', param_window_name, 50, 300, set_gamma)
    cv2.waitKey(0)


if hist_flag:
    image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (600, 600))
    histr = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histr)

    cv2.imshow("Lenna", image)

    plt.show()
    cv2.waitKey(0)

if eq_flag:
    img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (400, 400))
    eq_img = cv2.equalizeHist(img)

    histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    eq_histr = cv2.calcHist([eq_img], [0], None, [256], [0, 256])
    plt.plot(histr, label='Source')
    plt.plot(eq_histr, label='Equalized')
    plt.legend()

    cv2.imshow('Source image', img)
    cv2.imshow('Equalized Image', eq_img)
    plt.show()
    cv2.waitKey(0)