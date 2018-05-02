cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 14)
project(LKOFlow)

set(OpenCV_DIR /usr/local/opencv3/share/OpenCV)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})

add_executable(LKOFlow main.cpp LKOFlow.hpp Utils.hpp)

target_link_libraries(LKOFlow ${OpenCV_LIBS})