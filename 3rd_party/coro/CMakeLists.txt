cmake_minimum_required(VERSION 2.8)

file(GLOB FILES *.cpp)
if (WIN32)
	list(REMOVE_ITEM FILES ${CMAKE_CURRENT_SOURCE_DIR}/FiberLinux.cpp)
else()
	list(REMOVE_ITEM FILES ${CMAKE_CURRENT_SOURCE_DIR}/FiberWindows.cpp)
endif()

if (APPLE)
	add_definitions(-D_XOPEN_SOURCE)
endif()

add_library(coro STATIC ${FILES})
add_subdirectory(tests)
