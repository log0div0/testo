cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(license STATIC ${SRCS})
target_link_libraries(license stdc++fs tweetnacl)
add_executable(license_demo main.cpp)
target_link_libraries(license_demo license)
