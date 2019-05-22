cmake_minimum_required(VERSION 3.0)

if (WIN32)
	add_definitions(-DCOBJMACROS -DCINTERFACE -D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DVBOX_WITH_GLUE)
endif()

file(GLOB_RECURSE SRCS *.cpp)
add_library(vbox STATIC ${SRCS})
