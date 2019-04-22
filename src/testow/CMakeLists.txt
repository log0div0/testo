cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_SOURCE_DIR}/3rd_party/imgui)
include_directories(${CMAKE_SOURCE_DIR}/3rd_party/darknet/include)

file(GLOB SRCS *.cpp)

if(WIN32)
	add_compile_options("/U CINTERFACE")
endif()

add_executable(testow ${SRCS})
add_dependencies(testow qemu)
target_link_libraries(testow imgui qemu pugixml libdarknet testo-core)

if(WIN32)
	include_directories(${CMAKE_SOURCE_DIR}/3rd_party/darknet/include/windows)
	target_link_libraries(testow d3d11 d3dcompiler)
	file(GLOB OS_SPECIFIC_SRCS windows/*.cpp)
elseif(APPLE)
	file(GLOB OS_SPECIFIC_SRCS osx/*.cpp)
elseif(UNIX)
	target_link_libraries(testow glfw gl3w GL pthread dl stdc++fs)
	include_directories(${CMAKE_SOURCE_DIR}/3rd_party/gl3w)
	file(GLOB OS_SPECIFIC_SRCS linux/*.cpp)
else()
	message(FATAL_ERROR "Unknown OS")
endif()

target_sources(testow PUBLIC ${OS_SPECIFIC_SRCS})

if(GPU)
	target_link_libraries(testow cuda cudart cublas curand)
endif()
