import cv2
import numpy as np


def update():
    global thr
    lines = cv2.HoughLines(edges, 1, np.pi / 180, thr, np.array([]))
    # Draw lines on the image
    curr_img = img.copy()

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(curr_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Hough", curr_img)


def updateP():
    global thr_p, min_len_p
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=thr_p, minLineLength=min_len_p)
    # Draw lines on the image
    curr_img = img.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(curr_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("HoughP", curr_img)


def set_thr(val):
    global thr
    thr = val
    update()


def set_thr_p(val):
    global thr_p
    thr_p = val
    updateP()


def set_len_p(val):
    global min_len_p
    min_len_p = val
    updateP()


# Read image
img = cv2.imread('road.jpg', cv2.IMREAD_COLOR)
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)


thr = 250
thr_p = 10
min_len_p = 10

param_window_name = 'sliders'
cv2.namedWindow(param_window_name)
cv2.createTrackbar('thr', param_window_name, 250, 750, set_thr)
cv2.createTrackbar('thr p', param_window_name, 10, 100, set_thr_p)
cv2.createTrackbar('min len p', param_window_name, 10, 100, set_len_p)

#cv2.imshow("Hough", img)
#cv2.imshow("HoughP", img)
cv2.waitKey(0)
