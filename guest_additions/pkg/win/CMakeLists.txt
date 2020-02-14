cmake_minimum_required(VERSION 3.0)

add_executable(testo-guest-additions-helper testo-guest-additions-helper.cpp)
target_link_libraries(testo-guest-additions-helper process)
