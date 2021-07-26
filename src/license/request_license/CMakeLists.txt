cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)

add_executable(testo_request_license main.cpp)
target_link_libraries(testo_request_license CUDA::cudart_static license nn)

target_link_libraries(testo_request_license CUDA::cudart_static)
