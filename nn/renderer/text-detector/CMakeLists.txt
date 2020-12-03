cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(renderer-text-detector ${SRCS})
target_link_libraries(renderer-text-detector stb psf stdc++fs pthread)
