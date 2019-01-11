cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/GL/*.c)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(gl3w STATIC ${SRCS})
