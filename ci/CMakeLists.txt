cmake_minimum_required(VERSION 3.0.0)

if (CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-D_DEBUG)
endif()

file(GLOB_RECURSE SRCS *.cpp)

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/out/include
)

add_executable(tests ${SRCS})
add_dependencies(tests testo-core)
target_link_libraries(tests testo-core)
add_custom_target(run-tests
	COMMAND ${CMAKE_BINARY_DIR}/out/sbin/tests
	DEPENDS tests
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
