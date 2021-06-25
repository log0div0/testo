cmake_minimum_required(VERSION 3.0)

file(GLOB SRCS *.cpp)

add_definitions(-DNN_SERVICE_PROCOTOL_VERSION="1.0")
add_executable(nn_service ${SRCS})

target_link_libraries(nn_service coro stb)


if (UNIX)
	target_link_libraries(nn_service pthread)
endif()

add_subdirectory(client_demo)