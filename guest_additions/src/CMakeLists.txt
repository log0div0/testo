cmake_minimum_required(VERSION 3.0)

if (UNIX)
	set(CMAKE_EXE_LINKER_FLAGS "-static")
	set(SRCS
		Server.cpp
		linux/Channel.cpp
		linux/Process.cpp
		linux/main.cpp
	)
	add_executable(testo-guest-additions ${SRCS})
	target_link_libraries(testo-guest-additions stdc++fs)
endif()

if (WIN32)
	add_executable(testo-guest-additions win/main.cpp)
endif()
