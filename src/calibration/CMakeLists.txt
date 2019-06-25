cmake_minimum_required(VERSION 3.0.0)
project(opencv_sample VERSION 0.1.0)

add_executable(calibration 
  calibration.cc
  corner_detection_util.cc
  fisheye.cc
)

target_link_libraries(calibration PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_LIBS}
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  fileutil
)

add_executable(blob_corner_detection_check 
  blob_corner_detection_check.cc
  corner_detection_util.cc)

target_link_libraries(blob_corner_detection_check
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  fileutil
)

