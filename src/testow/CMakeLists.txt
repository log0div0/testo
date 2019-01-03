cmake_minimum_required(VERSION 3.0)

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party/imgui
)

file(GLOB_RECURSE SRCS *.cpp)

if(WIN32)
	add_compile_options("/U CINTERFACE")
	add_executable(testow ${SRCS})
	target_link_libraries(testow imgui d3d11 d3dcompiler vbox vboxcapi)
endif()
