cmake_minimum_required(VERSION 3.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/sbin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
	add_definitions(-O2 -DNDEBUG)
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-O0 -D_DEBUG -g)
	set(EXTRA_CFLAGS "-O0 -D_DEBUG -g")
else()
	MESSAGE(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE value: ${CMAKE_BUILD_TYPE}")
endif()

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party
)

add_subdirectory(3rd_party)
add_subdirectory(src)
