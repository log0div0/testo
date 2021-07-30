cmake_minimum_required(VERSION 3.0)

add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")
add_definitions(-DTESTO_RELEASE_DATE="${TESTO_RELEASE_DATE}")

find_package(CUDAToolkit)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

file(GLOB SRCS *.cpp)

add_executable(nn_service ${SRCS})

target_link_libraries(nn_service coro stb js nn license)
target_link_libraries(nn_service CUDA::cudart_static)

if (UNIX)
	target_link_libraries(nn_service pthread)
endif()

add_subdirectory(pkg)
add_subdirectory(client_demo)