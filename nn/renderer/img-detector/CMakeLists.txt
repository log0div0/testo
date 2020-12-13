cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(renderer-img-detector ${SRCS})
target_link_libraries(renderer-img-detector stb psf pthread)
