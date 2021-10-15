import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plot_img_table(imgs, names):
    nrows = imgs.shape[0]
    ncols = imgs.shape[1]
    fig, ax = plt.subplots(nrows, ncols, figsize=(7, 7))
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].imshow(imgs[i, j], cmap='gray')
            ax[i, j].title.set_text(names[i, j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()


img = cv.imread('lenna.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (400, 400))

kernel_size = 5

# blur
gauss_blur = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
median_blur = cv.medianBlur(img, kernel_size)

kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
kernel /= (kernel_size * kernel_size)
normalized_blur = cv.filter2D(img, -1, kernel)

blur_names = np.array([['Original', 'Gaussian'], ['Median', 'Normalized']])
blur_img = np.array([[img, gauss_blur], [median_blur, normalized_blur]])
plot_img_table(blur_img, blur_names)

# noise
noise = img.copy()
cv.randn(noise, 0, 32)
#cv.randu(noise, -48, 48)
_img = img + noise

kernel_size = 9
gauss_blur = cv.GaussianBlur(_img, (kernel_size, kernel_size), 0)
median_blur = cv.medianBlur(_img, kernel_size)
normalized_blur = cv.filter2D(_img, -1, kernel)

filter_img = np.array([[_img, gauss_blur], [median_blur, normalized_blur]])
filter_names = np.array([['Noisy', 'Gaussian'], ['Median', 'Normalized']])
plot_img_table(filter_img, filter_names)

# Sobel
sobel_x = cv.Sobel(img, -1, 0, 1)
sobel_y = cv.Sobel(img, -1, 1, 0)
sobel = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
laplacian = cv.Laplacian(img, -1)

edge_img = np.array([[sobel, laplacian], [sobel_x, sobel_y]])
edge_names = np.array([['Sobel', 'Laplacian'], ['Sobel_X', 'Sobel_Y']])
plot_img_table(edge_img, edge_names)

plt.show()
cv.waitKey(0)
