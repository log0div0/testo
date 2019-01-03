cmake_minimum_required(VERSION 3.0)

add_subdirectory(VBoxCAPI)
add_subdirectory(darknet)
add_subdirectory(imgui)

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
