cmake_minimum_required(VERSION 3.0.0)
project(opencv_sample VERSION 0.1.0)

file(GLOB GLOB_SOURCE ./*.cc)

add_library(fileutil ${GLOB_SOURCE})

target_include_directories(fileutil PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(fileutil
  ${GLOG_LIBRARIES}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)