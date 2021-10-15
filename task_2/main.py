import cv2
import numpy as np

img = cv2.imread('mike.jpg')

scale = 0.5
w = int(img.shape[1] * scale)
h = int(img.shape[0] * scale)

img = cv2.resize(img, (w, h))

cv2.circle(img, (w//2, int(5/16*h)), 20, (0, 0, 255), thickness=10)
cv2.rectangle(img, (int(2/16*w), int(10/16*h)), (int(4/16*w), int(8/16*h)), (255, 0, 0), thickness=5)

a = np.array([[180, 180], [300, 220], [200, 300]])
cv2.fillPoly(img, pts=[a], color=(0, 255, 0))

cv2.ellipse(img, (450, 80), (70, 40), -45, 90, 270, (255, 100, 150), thickness=3)
cv2.ellipse(img, (450, 80), (30, 40), -45, 90, 270, (255, 100, 150), thickness=3)

cv2.imwrite('mike_colored.jpg', img)
cv2.imshow('mike', img)
cv2.waitKey(0)