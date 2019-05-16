cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(hyperv ${SRCS})
target_link_libraries(hyperv wbemuuid)