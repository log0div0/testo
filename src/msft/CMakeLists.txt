cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(msft STATIC ${SRCS})
target_link_libraries(msft wbemuuid Propsys)
add_executable(msft_demo main.cpp)
target_link_libraries(msft_demo msft)
