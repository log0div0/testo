cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_library(quickjs-c++ STATIC ${SRCS})

target_link_libraries(quickjs-c++ quickjs)
