cmake_minimum_required(VERSION 3.0)

add_definitions(-DFMT_HEADER_ONLY)

if (WIN32)
	add_definitions(-D__PRETTY_FUNCTION__=__FUNCSIG__)
endif()

if (USE_CUDA)
	add_definitions(-DUSE_CUDA=1)
endif()

if (APPLE)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
endif()

include_directories(${ONNX_RUNTIME_DIR}/include/onnxruntime/core/session/)

if (USE_CUDA)
	include_directories(${ONNX_RUNTIME_DIR}/include/onnxruntime/core/providers/cuda/)
endif()

link_directories(${ONNX_RUNTIME_DIR}/lib/)

add_subdirectory(license)
add_subdirectory(js)
add_subdirectory(nn)
add_subdirectory(testo)
