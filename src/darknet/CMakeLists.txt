cmake_minimum_required(VERSION 3.0)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/out/include/openblas
)

file(GLOB_RECURSE libdarknet_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM libdarknet_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

if(GPU)
	file(GLOB_RECURSE cuda_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
	set(libdarknet_SRCS ${cuda_files} ${libdarknet_SRCS})
else()
	list(REMOVE_ITEM libdarknet_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda.c)
endif()

add_library(libdarknet STATIC ${libdarknet_SRCS})
set_target_properties(libdarknet PROPERTIES PREFIX "")
target_link_libraries(libdarknet clblast)

file(GLOB_RECURSE darknet_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_executable(darknet ${darknet_SRCS})
target_link_libraries(darknet libdarknet)
if(UNIX)
	target_link_libraries(darknet pthread m)
endif()
if(GPU)
	target_link_libraries(darknet cuda cudart cublas curand)
endif()
