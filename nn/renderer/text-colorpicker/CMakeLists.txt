cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(renderer-text-colorpicker ${SRCS})
target_link_libraries(renderer-text-colorpicker stb psf pthread)
