cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_SOURCE_DIR}/3rd_party/imgui)

file(GLOB SRCS *.cpp)

if (WIN32)
	add_compile_options("/U CINTERFACE")
endif()

add_executable(testow ${SRCS})
target_link_libraries(testow imgui vbox vboxcapi)

if(WIN32)
	target_link_libraries(testow d3d11 d3dcompiler)
	file(GLOB OS_SPECIFIC_SRCS windows/*.cpp)
elseif(APPLE)
	message(FATAL_ERROR "Implement me")
elseif(UNIX)
	target_link_libraries(testow glfw gl3w GL pthread dl)
	include_directories(${CMAKE_SOURCE_DIR}/3rd_party/gl3w)
	file(GLOB OS_SPECIFIC_SRCS linux/*.cpp)
else()
	message(FATAL_ERROR "Unknown OS")
endif()

target_sources(testow PUBLIC ${OS_SPECIFIC_SRCS})
