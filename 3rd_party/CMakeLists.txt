cmake_minimum_required(VERSION 3.0)

add_subdirectory(coro)
add_subdirectory(VBoxCAPI)
add_subdirectory(darknet)
add_subdirectory(imgui)

if(UNIX AND NOT APPLE)
add_subdirectory(gl3w)
endif()

include(ExternalProject)

ExternalProject_Add(
	json
	PREFIX ${CMAKE_BINARY_DIR}
	UPDATE_COMMAND ""
	# GIT_REPOSITORY https://github.com/nlohmann/json
	URL file://${CMAKE_CURRENT_SOURCE_DIR}/json-2.1.1.tar.gz
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory include ${CMAKE_BINARY_DIR}/out/include
)

ExternalProject_Add(
	libpugixml
	PREFIX ${CMAKE_BINARY_DIR}
	UPDATE_COMMAND ""
	URL file://${CMAKE_CURRENT_SOURCE_DIR}/pugixml-1.9.tar.gz
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND cmake .
	BUILD_COMMAND make
	INSTALL_COMMAND mkdir -p ${CMAKE_BINARY_DIR}/out/include/pugixml/ &&
		cp src/pugixml.hpp ${CMAKE_BINARY_DIR}/out/include/pugixml/ &&
		cp src/pugiconfig.hpp ${CMAKE_BINARY_DIR}/out/include/pugixml/ &&
		cp libpugixml.a ${CMAKE_BINARY_DIR}/out/lib
)

ExternalProject_Add(
	asio
	PREFIX ${CMAKE_BINARY_DIR}
	UPDATE_COMMAND ""
	# GIT_REPOSITORY https://github.com/chriskohlhoff/asio
	URL file://${CMAKE_CURRENT_SOURCE_DIR}/asio.tar.xz
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND mkdir -p ${CMAKE_BINARY_DIR}/out/include/asio
		&& cp -r include/asio ${CMAKE_BINARY_DIR}/out/include
		&& cp -r include/asio.hpp ${CMAKE_BINARY_DIR}/out/include
)
