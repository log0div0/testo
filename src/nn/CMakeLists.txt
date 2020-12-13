cmake_minimum_required(VERSION 3.0)

file(GLOB SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_library(nn STATIC ${SRCS})
target_link_libraries(nn stb onnxruntime)
if(UNIX)
	target_link_libraries(nn pthread)
endif()

add_executable(nn_demo main.cpp)
target_link_libraries(nn_demo nn)
