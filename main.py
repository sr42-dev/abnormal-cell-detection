'''

Notes :

- Pick up an image with abnormal cells [DONE]
- Make a CV algorithm to count the number of cells [DONE]
- Make a CV algorithm to count the number of abnormal cells [DONE]
- Document algorithm flowcharts [DONE]
- Prepare a prototype [DONE]

'''

# library imports
import cv2
import numpy as np

# image import
img = cv2.imread('resources/cancerCells.jpeg')

# algorithm for cell counting
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # conversion to HSV from BGR

# color filtering
lower = np.array([0, 0, 0])  # minimum range array
upper = np.array([179, 255, 225])  # maximum range array
mask = cv2.inRange(imgHSV, lower, upper)  # filtering out colours from HSV image

# cell counting from original image
imgCenters = cv2.imread('resources/cancerCells.jpeg')
ret, thresh = cv2.threshold(mask, 254, 255, 0)
eroded = cv2.erode(thresh, None, iterations=1)
dilated = cv2.dilate(eroded, None, iterations=1)

contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
countCells = 0

for c in contours:

    area = cv2.contourArea(c)
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # print(area)
    if (area > 10) and (area < 7000): # filtering for reasonably sized cells
        x, y, w, h = cv2.boundingRect(c)
        imageFrame = cv2.rectangle(imgCenters, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(imgCenters, (cX, cY), 3, (0, 255, 255), -1)
        countCells += 1

cv2.putText(imgCenters,'Total cell count : {}'.format(countCells), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, 2)

# algorithm for abnormal cell counting
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # conversion to HSV from BGR

# color filtering
lower = np.array([115 , 69 , 107])  # minimum range array
upper = np.array([145 , 192 , 255])  # maximum range array
mask = cv2.inRange(imgHSV, lower, upper)  # filtering out colours from HSV image

imgCentersAbnormal = cv2.imread('resources/cancerCells.jpeg')
ret, thresh = cv2.threshold(mask, 254, 255, 0)
eroded = cv2.erode(thresh, None, iterations=1)
dilated = cv2.dilate(eroded, None, iterations=1)

contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
countAbnormalCells = 0

for c in contours:

    area = cv2.contourArea(c)
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # print(area)
    if (area > 50) and (area < 7000): # filtering for reasonably sized cells
        x, y, w, h = cv2.boundingRect(c)
        imageFrame = cv2.rectangle(imgCentersAbnormal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(imgCentersAbnormal, (cX, cY), 3, (0, 255, 255), -1)
        countAbnormalCells += 1

cv2.putText(imgCentersAbnormal,'Abnormal cell count : {}'.format(countAbnormalCells), (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, 2)


# result display
cv2.imshow('OG image', img)
cv2.imshow('Cell count', imgCenters)
cv2.imshow('Abnormal cell count', imgCentersAbnormal)
print(countCells)
cv2.waitKey(0)