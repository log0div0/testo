cmake_minimum_required(VERSION 3.0)

file(GLOB SRCS *.cpp)

add_executable(nn_service_client_demo ${SRCS})

target_link_libraries(nn_service_client_demo coro stb)

if (UNIX)
	target_link_libraries(nn_service_client_demo pthread)
endif()
