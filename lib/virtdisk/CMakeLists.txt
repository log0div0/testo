cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(virtdisk++ STATIC ${SRCS})
target_link_libraries(virtdisk++ virtdisk)
add_executable(virtdisk_demo main.cpp)
target_link_libraries(virtdisk_demo virtdisk++)
