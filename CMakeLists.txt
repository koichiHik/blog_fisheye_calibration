cmake_minimum_required(VERSION 3.0.0)
project(opencv_sample VERSION 0.1.0)

set(CMAKE_CXX_FLAGS 
  ${CMAKE_CXX_FLAGS} "-Wall -std=c++11")

# DEPENDENCY
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem program_options)
find_package(yaml-cpp REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

link_directories(${GFLAGS_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_subdirectory(src/calibration)
add_subdirectory(src/fileutil)

