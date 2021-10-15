import cv2


def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) >= 5:
        ellipse = cv2.fitEllipse(approx)
        ratio = ellipse[1][0] / ellipse[1][1]
        ratio = ratio if ratio < 1 else (1 / ratio)
        shape = "circle" if ratio >= 0.9 else "ellipsis"
    # return the name of the shape
    return shape


im = cv2.imread('card.jpg')
img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

edged = cv2.Canny(img, 50, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


#ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_area = -1
S = im.shape[0] * im.shape[1]
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if max_area < area <= 0.95 * S:
        cnt = contours[i]
        max_area = area

print(max_area / S)
print(detect(cnt))

cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
cv2.imshow('contours', im)
cv2.waitKey(0)
