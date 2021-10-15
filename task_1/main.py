import cv2

img_color = cv2.imread('smeh.jfif')
img = cv2.imread('smeh.jfif', cv2.IMREAD_GRAYSCALE)

print("Type target: ")
target = int(input())

total = img.shape[0] * img.shape[1]
count = 0

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] >= target:
            count = count + 1

print("Percentage: ", count/total*100)


img_color = cv2.resize(img_color, (600, 300))
img = cv2.resize(img, (600, 300))

cv2.imshow('smeh_gray', img)
cv2.imshow('smeh_color', img_color)
cv2.waitKey(0)