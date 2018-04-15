cmake_minimum_required(VERSION 3.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/sbin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)

if (WIN32)
	add_definitions(-D__PRETTY_FUNCTION__=__FUNCSIG__ -DCOBJMACROS -DCINTERFACE -D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DVBOX_WITH_GLUE)
endif()

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party
)

add_subdirectory(src)
