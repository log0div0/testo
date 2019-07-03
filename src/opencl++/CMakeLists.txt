cmake_minimum_required(VERSION 3.0)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(opencl++ STATIC ${SRCS})
target_link_libraries(opencl++ ${OpenCL_LIBRARIES})
add_executable(opencl++_demo main.cpp)
target_link_libraries(opencl++_demo opencl++)
