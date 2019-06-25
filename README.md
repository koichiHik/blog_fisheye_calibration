# blog_fisheye_calibration

This is the sample code for fisheye calibration.
The code itself is extracted from opencv fisheye calibration and refactored by my self.

Little bit of code walk through is written in my blog.
http://daily-tech.hatenablog.com/entry/2019/06/10/030934

### Prerequisite
You need the following libraries for successful build.

OpenCV
GLOG
GFLAG
Boost
Eigen3

### Command

#### Build command
sh build.sh

#### Run command
##### 1. Checking detected corner.
./build/src/calibration/blob_corner_detection_check --flagfile ./src/calibration/blob_corner_detection_check_flags.txt

##### 2. Run calibration.
./build/src/calibration/calibration --flagfile ./src/calibration/calibration_flags.txt
