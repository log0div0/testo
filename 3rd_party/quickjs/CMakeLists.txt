cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

file(GLOB FILES *.c)

set(EXTRA_FLAGS "-O3 -D_GNU_SOURCE -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")
add_library(quickjs STATIC ${FILES})
