cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(TestOpenCV)

# Указываем, где искать собранную библиотеку
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv_ready")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(my_app 
main.cpp
ProcessedVideo.cpp      
FrameParametres.cpp     
TestingMetods.cpp       
ProcessingMethods.cpp)

target_link_libraries(my_app ${OpenCV_LIBS})