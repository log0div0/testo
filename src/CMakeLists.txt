cmake_minimum_required(VERSION 3.0)

add_definitions(-DFMT_HEADER_ONLY)
add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")
if (WIN32)
	add_definitions(
		-D__PRETTY_FUNCTION__=__FUNCSIG__
		/W1
		/WX
	)
endif()

if (USE_CUDA)
	add_definitions(-DUSE_CUDA=1)
endif()


if (APPLE)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
endif()

add_subdirectory(vbox)
add_subdirectory(process)
add_subdirectory(license)
if (WIN32)
add_subdirectory(virtdisk)
add_subdirectory(msft)
add_subdirectory(hyperv)
endif()
if (UNIX AND NOT APPLE)
add_subdirectory(qemu)
endif()
add_subdirectory(stb)

if (ONNX_RUNTIME_DIR)
	include_directories(
		${ONNX_RUNTIME_DIR}/include
	)

	link_directories(
		${ONNX_RUNTIME_DIR}/lib/
	)

	add_subdirectory(js)
	add_subdirectory(nn)
	add_subdirectory(testo)
endif()
