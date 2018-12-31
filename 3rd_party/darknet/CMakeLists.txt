cmake_minimum_required(VERSION 3.0)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(WIN32)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/windows)
endif()

file(GLOB_RECURSE libdarknet_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

if(GPU)
	add_definitions(-DGPU)
	enable_language("CUDA")
	file(GLOB_RECURSE cuda_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
	set(libdarknet_SRCS ${cuda_files} ${libdarknet_SRCS})
endif()

add_library(libdarknet STATIC ${libdarknet_SRCS})
set_target_properties(libdarknet PROPERTIES PREFIX "")

file(GLOB_RECURSE darknet_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.c)
add_executable(darknet ${darknet_SRCS})
target_link_libraries(darknet libdarknet)
if(UNIX)
	target_link_libraries(darknet pthread m)
endif()
if(GPU)
	target_link_libraries(darknet cuda cudart cublas curand)
endif()

add_custom_target(dataset
	COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/dataset.py
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
