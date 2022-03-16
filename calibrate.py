import numpy as np
import cv2
import glob
import os
import argparse
from loguru import logger
import json
from numpyencoder import NumpyEncoder


class calibration:
    def __init__(self, width, height):
        '''
        Define the number of corner in width and height
        '''
        self.width = width
        self.height = height

    def get_calibration(self, checker_dir: str, format: str, save):
        '''
        Compute calibration matrix

        checker_dir : location of checkerboard images
        format : format of checkerboard images

        '''
        assert format in ['jpg', 'jpeg', 'png', 'gif', 'tiff']

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        winSize = (11, 11)
        # Half of the side length of the search window. For example, if winSize=Size(5,5) , then a (5∗2+1)×(5∗2+1)=11×11 search window is used.
        zeroZone = (-1, -1)
        # Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. 
        # It is used sometimes to avoid possible singularities of the autocorrelation matrix. 
        # The value of (-1,-1) indicates that there is no such a size.

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.height*self.width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.height, 0:self.width].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob(checker_dir + '/*.' + format)

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray, (self.height, self.width), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, winSize, zeroZone, criteria)  # Increase the accuracy
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(
                    img, (self.height, self.width), corners2, ret)
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                cv2.imshow('img', img)
                cv2.waitKey()
            cv2.destroyAllWindows()

        ''' Calibration '''
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret < 1:
            logger.info(
                f"Your RMS re-projection error is {ret}. This is tolerable.")
        else:
            logger.info(
                f"Your RMS re-projection error is {ret}. Too high. Use the better quality of checker board images.")

        if save:
            name = checker_dir.split('/')[-1]
            np.savez(os.path.join(
                save, f'{name}.npz'), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            logger.info(
                f"Numpy array is saved at {save}. This file can be used for distortion.")
            result = {'Reprojection Error in RMS': ret, 'Intrinsic Parameter': mtx, 'Distortion Error': dist,
                      'Extrinsic Parameter': {'Rotational': rvecs, 'Translational': tvecs}}
            with open(os.path.join(save, f'{name}.json'), 'w') as f:
                json.dump(result, f, indent=4, cls=NumpyEncoder)
            logger.info(
                f"Json is saved at {save}.")

        return mtx, dist

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Get Camera Calibration Parameters")

    parser.add_argument("--checker",
                        default='checker/dev-0',
                        help="Location of checker board images .")

    parser.add_argument("--format",
                        default='jpg',
                        help="Format of your checker images.")

    parser.add_argument("--save",
                        default=False,
                        help="Location of calibration parameters to save")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    checker_dir = args.checker
    cal_format = args.format
    save = args.save

    cal = calibration(width=9, height=6)
    mtx, dist = cal.get_calibration(
        checker_dir=checker_dir, format=cal_format, save=save)
    print("matrice: \n", mtx)
    print("distortion coefficient: \n", dist)
