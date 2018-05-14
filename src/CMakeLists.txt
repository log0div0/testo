cmake_minimum_required(VERSION 3.0)

add_definitions(-std=gnu++14)

file(GLOB_RECURSE SRCS *.cpp)

include_directories(${CMAKE_BINARY_DIR}/out/include)
link_directories(${CMAKE_BINARY_DIR}/out/lib)

add_executable(testo ${SRCS})
add_dependencies(testo sdl2)

target_link_libraries(testo vboxcapi SDL2)
if (UNIX)
	target_link_libraries(testo dl pthread)
endif()
