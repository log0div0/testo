cmake_minimum_required(VERSION 3.0)

add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")
add_definitions(-DTESTO_RELEASE_DATE="${TESTO_RELEASE_DATE}")

include_directories(${ONNX_RUNTIME_DIR}/include/onnxruntime/core/session/)
link_directories(${ONNX_RUNTIME_DIR}/lib/)

if (USE_CUDA)
	find_package(CUDAToolkit)
	include_directories(${CUDAToolkit_INCLUDE_DIRS})
	add_definitions(-DUSE_CUDA=1)
	include_directories(${ONNX_RUNTIME_DIR}/include/onnxruntime/core/providers/cuda/)
endif()

file(GLOB SRCS *.cpp)

list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/client_demo.cpp)

if (UNIX)
	list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/MainWin.cpp)
endif()
if (WIN32)
	list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/MainLinux.cpp)
endif()

add_executable(testo_nn_service ${SRCS})

target_link_libraries(testo_nn_service coro stb js nn license)

if (USE_CUDA)
	target_link_libraries(testo_nn_service CUDA::cudart_static)
endif()

add_executable(testo_nn_service_client_demo client_demo.cpp)
target_link_libraries(testo_nn_service_client_demo stb coro)

if (UNIX)
	target_link_libraries(testo_nn_service pthread)
	target_link_libraries(testo_nn_service_client_demo pthread)
endif()

add_subdirectory(license)
add_subdirectory(js)
add_subdirectory(nn)
add_subdirectory(pkg)
