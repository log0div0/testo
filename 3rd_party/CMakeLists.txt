cmake_minimum_required(VERSION 3.0)

include(ExternalProject)
include(ProcessorCount)

ProcessorCount(N)

add_subdirectory(coro)
add_subdirectory(VBoxCAPI)
add_subdirectory(imgui)
add_subdirectory(pugixml)

if(WIN32)
ExternalProject_Add(
	libopenblas
	PREFIX ${CMAKE_BINARY_DIR}
	UPDATE_COMMAND ""
	URL file://${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS-develop.zip
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND cmake -G ${CMAKE_GENERATOR} -DUSE_THREAD=off -DNOFORTRAN=on -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/out
	BUILD_COMMAND cmake --build . --target INSTALL --config Release
	INSTALL_COMMAND ""
)
else()
ExternalProject_Add(
	libopenblas
	PREFIX ${CMAKE_BINARY_DIR}
	UPDATE_COMMAND ""
	URL file://${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS-develop.zip
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND cmake -DUSE_THREAD=off -DNOFORTRAN=on -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/out
	BUILD_COMMAND make -j ${N}
)
endif()

if(UNIX AND NOT APPLE)
add_subdirectory(gl3w)
endif()
