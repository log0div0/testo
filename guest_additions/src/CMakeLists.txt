cmake_minimum_required(VERSION 3.0)

set(SRCS
	Server.cpp
	Channel.cpp
	File.cpp)

if (UNIX)
	set(CMAKE_EXE_LINKER_FLAGS "-static")
	list(APPEND SRCS linux/main.cpp)
	add_executable(testo-guest-additions ${SRCS})
	target_link_libraries(testo-guest-additions stdc++fs process)
endif()

if (WIN32)
	list(APPEND SRCS win/main.cpp)
	add_executable(testo-guest-additions ${SRCS})
	target_link_libraries(testo-guest-additions SetupAPI process)
endif()
