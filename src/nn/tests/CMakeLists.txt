cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(nn_tests ${SRCS})
target_link_libraries(nn_tests nn)

if(UNIX)
add_custom_target(run-nn-tests
	COMMAND ${CMAKE_BINARY_DIR}/out/sbin/nn_tests
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	DEPENDS nn_tests
)
endif()
