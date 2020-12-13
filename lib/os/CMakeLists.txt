cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_library(os STATIC ${SRCS})
if (WIN32)
	target_link_libraries(os winapi)
else()
	target_link_libraries(os posixapi)
endif()
