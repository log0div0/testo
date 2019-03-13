cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_library(qemu STATIC ${SRCS})

add_dependencies(qemu libpugixml)
target_link_libraries(qemu virt)
