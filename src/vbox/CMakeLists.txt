cmake_minimum_required(VERSION 3.0)

if (WIN32)
	add_definitions(-DCOBJMACROS -DCINTERFACE -D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DVBOX_WITH_GLUE)
endif()

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(vbox STATIC ${SRCS})
target_link_libraries(vbox vboxcapi)
add_executable(vbox_demo main.cpp)
target_link_libraries(vbox_demo vbox)