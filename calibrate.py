import numpy as np
import cv2
import glob
import os


class calibration:
    def __init__(self, width, height):
        ''' 
        Define the number of corner in width and height
        '''
        self.width = width
        self.height = height

    def get_calibration(self, checker_dir, format='jpeg', save=False):
        ''' 
        Compute calibration matrix 
        
        checker_dir : location of checkerboard images
        format : format of checkerboard images

        '''

        assert format in ['jpg', 'jpeg', 'png', 'gif', 'tiff']

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.height*self.width,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.height,0:self.width].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.


        images = glob.glob(checker_dir + '/*.' + format)

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.height, self.width),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) # Increase the accuracy
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (self.height, self.width), corners2,ret)
                cv2.imshow('img',img)
                cv2.waitKey()
            cv2.destroyAllWindows()

        ''' Calibration '''
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        if save:    
            cal_save_dir = 'calibration'
            np.savez(os.path.join(cal_save_dir,'dev.npz'), mtx=mtx, dist=dist)

        return mtx, dist

    def undistort(self, mtx, dist, input_dir, output_dir, format='jpeg'):
        ''' 
        Calibrate Images

        mtx : cameraMatrix
        dist : distortion coefficients
        '''

        assert format in ['jpg', 'jpeg', 'png', 'gif', 'tiff']

        images = glob.glob(input_dir + '/*.' + format)

        for fname in images:

            img = cv2.imread(fname)
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, fname.split('/')[-1]), dst)
        
        print('Undistortion Process is Done!')

def undistort_api(config, input_dir, output_dir, cal_format, conv_format):
    ''' 
    Calibrate Images

    config : numpy array of cameraMatrix and distortion coefficients
    '''

    assert conv_format in ['jpg', 'jpeg', 'png', 'gif', 'tiff']

    config = np.load(config)
    mtx = config['mtx']
    dist = config['dist']

    images = glob.glob(input_dir + '/*.' + cal_format)

    for fname in images:

        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_dir, fname.split('/')[-1]), dst)
    
    print('Undistortion Process is Done!')

if __name__ == '__main__':

    input_dir = 'input'
    output_dir = 'output'
    cal_format = 'jpeg'
    conv_format = 'jpeg'

    cal = calibration(width = 9, height = 6)
    mtx, dist = cal.get_calibration(checker_dir = 'checker', format= cal_format, save=False)
    cal.undistort(mtx, dist, input_dir, output_dir, format= conv_format)

    ## if you want to use the saved calibration matrix
    # undistort_api('calibration/dev-6.npz','input', 'output', cal_format='jpeg', conv_format='jpeg')
