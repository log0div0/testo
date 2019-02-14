cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/out/include
)

add_executable(qemu-test ${SRCS})
# add_dependencies(tests testo-core)
target_link_libraries(qemu-test virt)
