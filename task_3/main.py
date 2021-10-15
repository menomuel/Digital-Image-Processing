import cv2

img_color = cv2.imread('text.jpg')
img = cv2.imread('text.jpg', cv2.IMREAD_GRAYSCALE)

enhance_img_1 = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=10)
enhance_img_2 = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=9, C=10)

img = cv2.resize(img, (600, 300))

cv2.imshow('init_text', img)
cv2.imshow('proc_text_1', enhance_img_1)
cv2.imshow('proc_text_2', enhance_img_2)
cv2.waitKey(0)