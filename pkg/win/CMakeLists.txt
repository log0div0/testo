cmake_minimum_required(VERSION 3.0)

add_executable(testo-helper WIN32 testo-helper.cpp)
target_link_libraries(testo-helper winapi)
