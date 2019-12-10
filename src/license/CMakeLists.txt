cmake_minimum_required(VERSION 3.0)

if (APPLE)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
endif()

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(license STATIC ${SRCS})
target_link_libraries(license sodium)
add_executable(license_demo main.cpp)
target_link_libraries(license_demo license)
