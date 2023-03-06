import math

import numpy as np
import cv2
import glob

from EniPy import colors
from EniPy import eniUtils
from EniPy import zoom



def readCameraCalibrationData(filename):
    calibrationData = eniUtils.readJson(filename)

    calibrationData['mtx'] = np.array(calibrationData['mtx'])
    calibrationData['dist'] = np.array(calibrationData['dist'])

    return calibrationData
def undistort(img, calibrationData):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(calibrationData['mtx'], calibrationData['dist'], (w, h), 1, (w, h))
    dst = cv2.undistort(img, calibrationData['mtx'], calibrationData['dist'], None, newcameramtx)
    return dst
def saveCameraCalibrationData(filename):
    chessboardWidth = 7
    chessboardHeight = 7

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardWidth * chessboardHeight, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardHeight, 0:chessboardWidth].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('images/calibration/9_9/*.jpg')
    for fname in images:
        print(f'load {fname}')
        img = loadImage(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (chessboardHeight, chessboardWidth), None)
        # If found, add object points, image points (after refining them)
        print(f'result {ret}')
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (chessboardHeight, chessboardWidth), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(5000)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    result = dict()
    result['mtx'] = mtx
    result['dist'] = dist

    eniUtils.writeJson(filename, result)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

def loadImage(filename):
    targetWidth = 1920
    image = cv2.imread(filename)

    scale = targetWidth / image.shape[1]
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image

def getScaledImage(image, targetWidth = 1920):
    scale = targetWidth / image.shape[1]
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    scaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return scaled

def cropWorkZone(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_lower = np.array([111, 103, 67])
    color_upper = np.array([146, 242, 201])
    mask = cv2.inRange(hsv, color_lower, color_upper)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'cropWorkZone: found {len(contours)} contours')
    biggest = max(contours, key=lambda current: cv2.boundingRect(current)[2] * cv2.boundingRect(current)[3])

    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        color = colors.Green
        if(cnt is  biggest):
            color = colors.Red
        cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                      color, 1)
    cv2.imshow('cropCnt', image)

    x, y, w, h = cv2.boundingRect(biggest)
    result = image[y:y + h, x:x + w]

    return result

def calculateBoxParameters(points):
    for i in range(0, len(points)):
        p1Index = i
        p2Index = (p1Index + 1) % len(points)
        p1 = points[p1Index]
        p2 = points[p2Index]
        length = cv2.norm(p2 - p1)
        print(f'{p1Index}->{p2Index} = {length}')

    for i in range(0, len(points)):
        aIndex = i - 1
        if aIndex < 0:
            aIndex = aIndex + len(points)
        bIndex = i
        cIndex = (i + 1) % len(points)
        a = points[aIndex]
        b = points[bIndex]
        c = points[cIndex]

        ba = a - b
        bc = c - b

        dot = ba.dot(bc)

        cosAngle = dot / (cv2.norm(ba) * cv2.norm(bc))
        angle = math.degrees(math.acos(cosAngle))

        print(f'Angle {aIndex}->{bIndex}->{cIndex} = {angle}')


if __name__ == '__main__':
    saveCameraCalibrationData('calibration.json')

    cailbrationData = readCameraCalibrationData('calibration.json')

    original = loadImage('images/mats/h2_0.JPG')
    undist = undistort(original, cailbrationData)

    #cv2.imshow('undist', undist)

    cropped = cropWorkZone(undist)
    cv2.imshow('cropped', cropped)

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    # lower bound and upper bound for Green color
    color_lower = np.array([50, 100, 100])
    color_upper = np.array([75, 255, 255])
    mask = cv2.inRange(hsv, color_lower, color_upper)

    kernel = np.ones((13, 13), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'found {len(contours)} contours')
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.drawContours(result, contours, -1, colors.Blue, 3)
    biggest = max(contours, key=lambda current: cv2.boundingRect(current)[2] * cv2.boundingRect(current)[3])

    rect = cv2.boundingRect(biggest)
    print(f'bound rect at [{rect[0]};{rect[1]}] {rect[2]}x{rect[3]}')
    minRect = cv2.minAreaRect(biggest)

    print(f'min rect {minRect[1][0]}x{minRect[1][1]}')

    box = cv2.boxPoints(minRect)
    calculateBoxParameters(box)
    print(f'boxPoint at [{box[0][0]};{box[0][1]}]')
    box = np.intp(box)

    cv2.drawContours(result, [box], 0, colors.Red)

    # cv2.imshow('mask', mask)
    # cv2.imshow('closing', closing)
    #
    cv2.imshow('result', getScaledImage(result, 1200))
    zoom.PanZoomWindow(result, 'Result')

    cv2.waitKey()
    cv2.destroyAllWindows()
