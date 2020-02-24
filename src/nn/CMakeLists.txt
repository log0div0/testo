cmake_minimum_required(VERSION 3.0)

file(GLOB SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_library(nn STATIC ${SRCS})
target_link_libraries(nn stb)

set(CMAKE_EXE_LINKER_FLAGS "${ONNX_RUNTIME_LINKER_FLAGS}")

if(UNIX)
	target_link_libraries(nn pthread dl stdc++fs)
endif()

add_executable(nn_demo main.cpp)
target_link_libraries(nn_demo nn)

file(GLOB_RECURSE TEST_SRCS tests/*.cpp)
add_executable(nn_tests ${TEST_SRCS})
target_link_libraries(nn_tests nn)

if(UNIX)
add_custom_target(run-nn-tests
	COMMAND ${CMAKE_BINARY_DIR}/out/sbin/nn_tests --durations yes
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
	DEPENDS nn_tests
)
endif()
