cmake_minimum_required(VERSION 2.8)

file(GLOB FILES *.cpp)

add_executable(coro_tests ${FILES})
target_link_libraries(coro_tests coro)
if (UNIX)
	target_link_libraries(coro_tests pthread)
endif()
