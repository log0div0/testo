cmake_minimum_required(VERSION 3.0)

if (CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-D_DEBUG)
endif()

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_library(testo-core STATIC ${SRCS})

target_link_libraries(testo-core coro vbox vboxcapi qemu pugixml pthread)

if(UNIX)
	target_link_libraries(testo-core dl stdc++fs)
endif()

add_executable(testo main.cpp)
add_dependencies(testo testo-core)
target_link_libraries(testo testo-core)
