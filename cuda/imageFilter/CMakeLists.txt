cmake_minimum_required(VERSION 3.10)
project(ImageFilter CXX CUDA)



set(SOURCE_FILES
        GrayScale/grayscale.cu
        GrayScale/GrayScale.cpp
        Blur/blurImage.cpp
        Blur/blurcuda.cu
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 -std=c++14")

find_package(OpenCV REQUIRED)
add_executable(ImageFilter main.cpp ${SOURCE_FILES})

target_link_libraries(ImageFilter ${OpenCV_LIBS} )
