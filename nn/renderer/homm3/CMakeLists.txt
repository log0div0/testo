cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(renderer-homm3 ${SRCS})
target_link_libraries(renderer-homm3 stb psf pthread)
