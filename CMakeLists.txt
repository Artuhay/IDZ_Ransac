cmake_minimum_required(VERSION 3.14)
project(IDZ_Ransac)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 4.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(IDZ_Ransac main.cpp)

target_link_libraries(IDZ_Ransac ${OpenCV_LIBS})