cmake_minimum_required(VERSION 3.0)

add_definitions(-std=gnu++14)

file(GLOB_RECURSE SRCS *.cpp)

add_executable(testo ${SRCS})
target_link_libraries(testo VBoxCAPIGlue)
add_dependencies(testo VBoxCAPIGlue)
