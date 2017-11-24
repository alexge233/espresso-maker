cmake_minimum_required(VERSION 2.8)
project(espresso_maker)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)
set(VERSION_STRING ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

SET(EIGEN3_INCLUDE_DIR "/usr/include/eigen3/")
IF(NOT EIGEN3_INCLUDE_DIR)
    MESSAGE( FATAL_ERROR "Set EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 location")
ENDIF()
INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")
find_package(CUDA REQUIRED) 
include_directories(${CUDA_INCLUDE_DIRS})

# Caffe2_GPU
###
message(STATUS "Searching for libcaffe2_gpu.so")
find_library(Caffe_GPU_LIB NAMES Caffe2_GPU libCaffe2_GPU caffe2_gpu HINTS ${CAFFE_LIBDIR})
if (NOT Caffe_GPU_LIB)
    message(FATAL_ERROR "libcaffe2_gpu.so not found")
else()
    message(STATUS "libcaffe2_gpu.so found")
endif()
###
# Caffe2_CPU
###
message(STATUS "Searching for libcaffe2.so")
find_library(Caffe_CPU_LIB NAMES caffe2 Caffe2_CPU caffe2_cpu HINTS ${CAFFE_LIBDIR})
if (NOT Caffe_CPU_LIB)
    message(FATAL_ERROR "libcaffe2.so not found")
else()
    message(STATUS "libcaffe2.so found")
endif()

include_directories(${CAFFE_INCLUDES})

message(STATUS "Searching for OpenCV")
find_package(OpenCV 3.3 REQUIRED core 
                                 imgproc
                                 cudawarping
                                 cudaimgproc
                                 cudaobjdetect
                                 cudafeatures2d
                                 cudafilters
                                 cudaarithm
                                 )
include_directories(${OpenCV_INCLUDE_DIRS})
if (!${OpenCV_VERSION})
    message(ERROR "OpenCV required")
else()
    message(STATUS "OpenCV Found")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

set(LIBTARGET ${LIBTARGET} espresso-cuda)
add_library(${LIBTARGET} SHARED espresso.cpp)
set_target_properties(${LIBTARGET} PROPERTIES VERSION ${VERSION_STRING} 
                                              SOVERSION ${VERSION_STRING})
target_link_libraries(${LIBTARGET}                      ${CUDA_LIBRARIES}
                     ${OpenCV_LIBS}
                     ${Caffe_GPU_LIB}
                     ${Caffe_CPU_LIB}
                     ${ZBAR_LIBRARIES}
                     ${GLOG_LIB}
                     ${CURAND_LIBRARY}
                     )
set_target_properties(${LIBTARGET} 
                      PROPERTIES INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable(example example.cpp)
target_link_libraries(example ${LIBTARGET}
                     ${CUDA_LIBRARIES}
                     ${OpenCV_LIBS}
                     ${Caffe_GPU_LIB}
                     ${Caffe_CPU_LIB}
                     ${ZBAR_LIBRARIES}
                     ${GLOG_LIB}
                     ${CURAND_LIBRARY}
                     )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
set_property(TARGET ${LIBTARGET} PROPERTY CXX_STANDARD 14)


