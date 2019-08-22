cmake_minimum_required(VERSION 3.0)

add_custom_target(run-tests
	COMMAND SBIN_DIR=${CMAKE_BINARY_DIR}/out/sbin bats -r ${CMAKE_CURRENT_SOURCE_DIR}
	DEPENDS testo
)