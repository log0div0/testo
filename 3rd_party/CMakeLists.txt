cmake_minimum_required(VERSION 3.0)

add_subdirectory(VBoxCAPI)

include(ExternalProject)

ExternalProject_Add(
	sdl2
	URL file://${CMAKE_CURRENT_SOURCE_DIR}/SDL2-2.0.8.zip
	CMAKE_ARGS -DSDL_STATIC=on -DSDL_SHARED=off -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/out
)
