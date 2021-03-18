import cv2
import numpy as np
from math import inf
# image = cv2.imread(
#     "D:\\Studie\\Business Analytics\\Learning Machines\\Project\\assignment_2\\learning_machines_robobo\\src\\test_pictures.png"
# )


lower_green = np.array([0, 150, 0])
upper_green = np.array([20, 255, 20])
lower_red = np.array([0, 0, 100])
upper_red = np.array([90, 90, 255])
upper_color=upper_red
lower_color = lower_red
# mask = cv2.inRange(image, lower_green, upper_green)


def getContours(img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return -inf, -inf
    
    index_value, value = closestBlock(contours)
    cnt = contours[index_value]
    area = cv2.contourArea(cnt)
    # print(area)
    cv2.drawContours(img, cnt, -1, (255, 0, 0), 1)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return -inf, -inf
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
    return cX, value


def closestBlock(contourList):
    yVal = []
    for c in contourList:
        yVal.append(tuple(c[c[:, :, 1].argmax()][0])[1])
    return yVal.index(max(yVal)), max(yVal)


def checkSide(xCoordinate, img):
    height, width, dim = img.shape
    if 0 <= xCoordinate < width / 3:
        return 1
    elif width / 3 <= xCoordinate < width * 2 / 3:
        return 2
    elif width * 2 / 3 <= xCoordinate <= width:
        return 3




def selectHeading(img):
    # print('in select heading')
    mask = cv2.inRange(img, lower_color, upper_color)
    xCoord, value = getContours(mask)
    if xCoord != -inf:
        
        heading = checkSide(xCoord, img)
        
        # return heading
    else:
        heading = 0
    # print('in heading function', heading)
    return heading, value

# xCoord = getContours(mask)
# print(checkSide(xCoord, image))

# cv2.imshow("Image", image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
