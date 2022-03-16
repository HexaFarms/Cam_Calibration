import numpy as np
import cv2
import glob
import os
import argparse
from loguru import logger


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Distort images")

    parser.add_argument("--calib",
                        default='calibration/dev-0.npz',
                        help="Location of camera parameters.")

    parser.add_argument("--input",
                        default="input",
                        help="Location of image directory to be undistorted.")

    parser.add_argument("--output",
                        default="output",
                        help="Location of image directory to be saved.")

    args = parser.parse_args()
    return args


def undistort_api(config: str, input_dir: str, output_dir: str, conv_format='jpg', mode='fisheye'):
    ''' 
    Calibrate Images

    config : numpy array of cameraMatrix and distortion coefficients
    '''

    assert conv_format in ['jpg', 'jpeg', 'png', 'gif', 'tiff']

    config = np.load(config)
    mtx = config['mtx']
    dist = config['dist']

    images = glob.glob(input_dir + '/*.' + conv_format)
    logger.info(f"{len(images)} images are loaded.")

    for fname in images:

        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        # Returns the new camera intrinsic matrix based on the free scaling parameter.
        # alpha:Free scaling parameter between 0 (when all the pixels in the undistorted image are valid)
        #       and 1 (when all the source image pixels are retained in the undistorted image).

        # undistort
        if mode == 'fisheye':
            newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist[:,:-1].squeeze(), (w, h), np.eye(3), balance=0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K=mtx, D=dist[:,:-1].squeeze(), R=np.eye(3), P=newcameramtx, size=(w, h), m1type=cv2.CV_32FC1)
            dst = cv2.remap(
                img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        else:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                cameraMatrix=mtx, distCoeffs=dist, imageSize=(w, h), alpha=1, newImgSize=(w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(
                mtx, dist, None, newcameramtx, (w, h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

        # crop the image
        
        cv2.imwrite(os.path.join(output_dir, fname.split('/')[-1]), dst)

    logger.info('Undistortion Process is Done!')


if __name__ == '__main__':

    args = parse_args()

    config = args.calib
    input_dir = args.input
    output_dir = args.output

    undistort_api(config, input_dir, output_dir)
