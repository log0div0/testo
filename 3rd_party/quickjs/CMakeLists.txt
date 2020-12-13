cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

file(GLOB FILES *.c)

add_library(quickjs STATIC ${FILES})
