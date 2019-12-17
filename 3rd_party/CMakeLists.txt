cmake_minimum_required(VERSION 3.0)

include(ExternalProject)

add_subdirectory(coro)
add_subdirectory(VBoxCAPI)
add_subdirectory(pugixml)

ExternalProject_Add(
	spdlog
	PREFIX ${CMAKE_BINARY_DIR}
	UPDATE_COMMAND ""
	URL file://${CMAKE_CURRENT_SOURCE_DIR}/spdlog-1.4.2.tar.gz
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND mkdir -p ${CMAKE_BINARY_DIR}/out/include/spdlog
		&& cp -r include/spdlog ${CMAKE_BINARY_DIR}/out/include
)
