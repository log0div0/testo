cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(renderer-homm3-detector ${SRCS})
target_link_libraries(renderer-homm3-detector stb psf pthread)
