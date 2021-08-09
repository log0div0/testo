cmake_minimum_required(VERSION 3.0)

add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")
add_definitions(-DTESTO_RELEASE_DATE="${TESTO_RELEASE_DATE}")

if (USE_CUDA)
	find_package(CUDAToolkit)
	include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

file(GLOB SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/client_demo.cpp)

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

if (USE_CUDA)
	add_subdirectory(pkg)
endif()
