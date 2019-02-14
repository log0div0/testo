cmake_minimum_required(VERSION 3.0)

project(testo)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/sbin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)

if(GPU)
	add_definitions(-DGPU)
	enable_language("CUDA")
endif()

if(WIN32)
	set(EXTRA_FLAGS "/W0 -DNOMINMAX")
else()
	if(CMAKE_BUILD_TYPE MATCHES Release)
		set(EXTRA_FLAGS "${EXTRA_FLAGS} -Wall -Wextra -Wpedantic -O3 -DNDEBUG -g")
	elseif(CMAKE_BUILD_TYPE MATCHES Debug)
		set(EXTRA_FLAGS "${EXTRA_FLAGS} -O0 -D_DEBUG -g")
	else()
		MESSAGE(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE value: ${CMAKE_BUILD_TYPE}")
	endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(3rd_party)
add_subdirectory(ci)
add_subdirectory(src)
