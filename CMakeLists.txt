project(yolov7-opencv-dnn-cpp)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -pthread")
add_definitions(-std=c++14)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})

add_executable(yolo main.cpp yolo.cpp yolo.h)
target_link_libraries(yolo ${OpenCV_LIBRARIES})