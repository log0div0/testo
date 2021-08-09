cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/request_license.cpp)

if (USE_CUDA)
	find_package(CUDAToolkit)
	include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

add_library(license STATIC ${SRCS})
target_link_libraries(license tweetnacl)

if (USE_CUDA)
	add_executable(testo_request_license ${SRCS} request_license.cpp)
	target_link_libraries(testo_request_license CUDA::cudart_static license nn)
endif()

add_executable(testo_license_cgi main.cpp)
target_link_libraries(testo_license_cgi tweetnacl license)
