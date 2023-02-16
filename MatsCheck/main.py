import numpy as np
import cv2
import glob


def cameraCalibrate():
    chessboardWidth = 13
    chessboardHeight = 9

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardWidth * chessboardHeight, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardHeight, 0:chessboardWidth].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('images/calibration/1.jpg')
    for fname in images:
        print(f'load {fname}')
        img = cv2.imread(fname)
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
    pass

def loadImage(filename):
    targetWidth = 1080
    image = cv2.imread(filename)

    scale = targetWidth / image.shape[1]
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image

if __name__ == '__main__':
        original = loadImage('images/mats/a4.jpg')
        cv2.imshow('original', original)

        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        # lower bound and upper bound for Blue color
        color_lower = np.array([111, 103, 67])
        color_upper = np.array([146, 242, 201])
        # lower bound and upper bound for Green color
        # color_lower = np.array([50, 100, 100])
        # color_upper = np.array([75, 255, 255])
        mask = cv2.inRange(hsv, color_lower, color_upper)

        kernel = np.ones((13, 13), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f'found {len(contours)} contours')
        result = original.copy()
        cv2.drawContours(result, contours, -1, (255, 0, 0), 3)

        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            print(f'bound rect at [{rect[0]};{rect[1]}] {rect[2]}x{rect[3]}')
            minRect = cv2.minAreaRect(cnt)

            print(f'min rect {minRect[1][0]}x{minRect[1][1]}')

            box = cv2.boxPoints(minRect)
            print(f'boxPoint at [{box[0][0]};{box[0][1]}]')
            box = np.intp(box)
            cv2.drawContours(result, [box], 0, (0, 0, 255))

            cv2.rectangle(result, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                          (0, 255, 0), 1)

        cv2.imshow('mask', mask)
        cv2.imshow('closing', closing)

        cv2.imshow('result', result)

        cv2.waitKey()
        cv2.destroyAllWindows()
