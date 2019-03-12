cmake_minimum_required(VERSION 2.8)

add_definitions(-std=gnu++14)

file(GLOB_RECURSE FILES *.cpp)
list(REMOVE_ITEM FILES ${CMAKE_CURRENT_SOURCE_DIR}/FiberWindows.cpp)

add_library(coro STATIC ${FILES})
add_dependencies(coro asio)
