cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(quickjs++ STATIC ${SRCS})
target_link_libraries(quickjs++ quickjs)
add_executable(quickjs++_demo main.cpp)
target_link_libraries(quickjs++_demo quickjs++ nn)