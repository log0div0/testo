cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(stb STATIC ${SRCS})
add_executable(stb_demo main.cpp)
target_link_libraries(stb_demo stb)
