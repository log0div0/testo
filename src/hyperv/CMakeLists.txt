cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(hyperv STATIC ${SRCS})
target_link_libraries(hyperv wbemuuid)
add_executable(hyperv_demo main.cpp)
target_link_libraries(hyperv_demo hyperv)