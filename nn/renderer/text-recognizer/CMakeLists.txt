cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(renderer-text-recognizer ${SRCS})
target_link_libraries(renderer-text-recognizer stb psf pthread)
