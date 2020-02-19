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

include_directories(
	${ONNX_RUNTIME_DIR}/include
)

link_directories(
	${ONNX_RUNTIME_DIR}/lib/
)

set(ONNX_RUNTIME_LINKER_FLAGS " \
	-Wl,--whole-archive \
		-Wl,-lonnxruntime_common \
		-Wl,-lonnxruntime_framework \
		-Wl,-lonnxruntime_graph \
		-Wl,-lonnxruntime_mlas \
		-Wl,-lonnxruntime_optimizer \
		-Wl,-lonnxruntime_providers \
		-Wl,-lonnxruntime_session \
		-Wl,-lonnxruntime_util \
		-Wl,-lprotobuf-lite \
		-Wl,-lonnx \
		-Wl,-lonnx_proto \
		-Wl,-lre2 \
	-Wl,--no-whole-archive \
")

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
add_subdirectory(quickjs)
add_subdirectory(stb)
add_subdirectory(nn)
add_subdirectory(testo)
