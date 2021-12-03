# Camera Calibration Module
This repo consists of two parts to help camera calibration using OpenCV-python.


## 1. Get Callibration matrix

The number of inner corner is required.
The checker board which is provided in 'demo/pattern_chessboard.png' has 9 and 6 corners in width and height respectively.
Multiple checker board images with different angle could improve the accuracy.
<div align="center">
  <img src="demo\checker3.jpeg" width="500"/>
  <figcaption align = "center"><b>Fig.1 - Checkerboard</b></figcaption>
</div>
<br />

## 2. Undistort images using information of calibration.

By using the calibration matrix and distortion factors, images can be undistorted.
<div align="center">
  <img src="demo\test_before.jpeg" width="500"/> <figcaption align = "center"><b>Fig.2 - Original</b></figcaption>
</div>
<div align="center">
  <img src="demo\test_after.jpeg" width="500"/> 
  <figcaption align = "center"><b>Fig.3 - Undistorted</b></figcaption>

</div>
<br />