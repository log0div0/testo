cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(psf STATIC ${SRCS})
target_link_libraries(psf stb z)
add_executable(psf_demo main.cpp)
target_link_libraries(psf_demo psf)
