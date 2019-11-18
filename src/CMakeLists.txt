cmake_minimum_required(VERSION 3.0)

add_definitions(-DFMT_HEADER_ONLY)
if (WIN32)
	add_definitions(
		-D__PRETTY_FUNCTION__=__FUNCSIG__
		/W1
		/WX
	)
endif()

if(NOT ONNX_RUNTIME_DIR)
	message(FATAL_ERROR "ONNX_RUNTIME_DIR is not set")
endif()

link_directories(
	${ONNX_RUNTIME_DIR}/lib/
)

add_subdirectory(vbox)
if (WIN32)
add_subdirectory(virtdisk)
add_subdirectory(msft)
add_subdirectory(hyperv)
endif()
if (UNIX AND NOT APPLE)
add_subdirectory(qemu)
endif()
add_subdirectory(stb)
add_subdirectory(nn)
add_subdirectory(testo)
