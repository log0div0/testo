cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

file(GLOB FILES *.c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
add_library(quickjs STATIC ${FILES})
