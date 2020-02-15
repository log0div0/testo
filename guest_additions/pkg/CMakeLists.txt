cmake_minimum_required(VERSION 3.0)

if (UNIX)
	add_subdirectory(linux)
endif()

if (WIN32)
	add_subdirectory(win)
endif()
