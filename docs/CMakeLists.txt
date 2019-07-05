cmake_minimum_required(VERSION 3.0)

add_custom_target(docs
	COMMAND
		sphinx-autobuild ${CMAKE_CURRENT_SOURCE_DIR}/source/ ${CMAKE_BINARY_DIR}/out/docs
)