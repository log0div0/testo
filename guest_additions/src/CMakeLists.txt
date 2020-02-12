cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)

set(LIBS
	stdc++fs
	)

set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(testo-guest-additions ${SRCS})
target_link_libraries(testo-guest-additions ${LIBS})
