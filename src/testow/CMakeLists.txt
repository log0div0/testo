cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_SOURCE_DIR}/3rd_party/imgui)

if(NOT TENSORFLOW_DIR)
	message(FATAL_ERROR "TENSORFLOW_DIR is not set")
endif()

include_directories(
	${TENSORFLOW_DIR}
	${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include
)

link_directories(
	${TENSORFLOW_DIR}/tensorflow/lite/tools/make/gen/linux_x86_64/lib/
)

file(GLOB SRCS *.cpp)

add_executable(testow ${SRCS})
target_link_libraries(testow imgui pugixml testo-core)

if(WIN32)
	target_link_libraries(testow d3d11 d3dcompiler)
	file(GLOB OS_SPECIFIC_SRCS windows/*.cpp)
elseif(APPLE)
	file(GLOB OS_SPECIFIC_SRCS osx/*.cpp)
elseif(UNIX)
	target_link_libraries(testow glfw gl3w GL pthread dl qemu stdc++fs)
	include_directories(${CMAKE_SOURCE_DIR}/3rd_party/gl3w)
	file(GLOB OS_SPECIFIC_SRCS linux/*.cpp)
else()
	message(FATAL_ERROR "Unknown OS")
endif()

target_sources(testow PUBLIC ${OS_SPECIFIC_SRCS})
