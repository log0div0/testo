cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
link_directories(${CMAKE_BINARY_DIR}/out/lib)
add_executable(testo ${SRCS})

target_link_libraries(testo vboxcapi dl stdc++fs lept tesseract)
add_definitions(-std=c++17 -Wall)
