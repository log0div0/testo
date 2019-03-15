cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_library(qemu STATIC ${SRCS})

target_link_libraries(qemu virt pugixml)
