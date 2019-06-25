#!/bin/bash

#ROOT_3RD=~/workspace/3rdParty
CMAKE_BUILD_TYPE=Debug
#OpenCV_DIR=${ROOT_3RD}/opencv331/installd/share/OpenCV
#Eigen3_DIR=${ROOT_3RD}/eigen334/install/share/eigen3/cmake/

# GLOGS
GLOG_INCLUDE_DIRS=/usr/include
GLOG_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu/
GLOG_LIBRARIES=glog

# GFLAGS
GFLAGS_INCLUDE_DIRS=/usr/include
GFLAGS_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu/
GFLAGS_LIBRARIES=gflags

if [ ! -e ./build ]; then
  mkdir build
fi
cd build

cmake \
  -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  -D GLOG_INCLUDE_DIRS=${GLOG_INCLUDE_DIRS} \
  -D GLOG_LIBRARY_DIRS=${GLOG_LIBRARY_DIRS} \
  -D GLOG_LIBRARIES=${GLOG_LIBRARIES} \
  -D GFLAGS_INCLUDE_DIRS=${GFLAGS_INCLUDE_DIRS} \
  -D GFLAGS_LIBRARY_DIRS=${GFLAGS_LIBRARY_DIRS} \
  -D GFLAGS_LIBRARIES=${GFLAGS_LIBRARIES} \
  ../

make

cd ../
