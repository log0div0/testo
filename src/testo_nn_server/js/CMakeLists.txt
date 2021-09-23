cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(js STATIC ${SRCS})
target_link_libraries(js quickjs coro)
add_executable(js_demo main.cpp)
target_link_libraries(js_demo js nn)
