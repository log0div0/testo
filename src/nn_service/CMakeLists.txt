cmake_minimum_required(VERSION 3.0)

file(GLOB SRCS *.cpp)

add_executable(nn_service ${SRCS})

target_link_libraries(nn_service coro stb nn)

if (UNIX)
	target_link_libraries(nn_service pthread)
endif()

add_subdirectory(client_demo)