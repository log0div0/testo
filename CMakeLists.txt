cmake_minimum_required(VERSION 3.0)

project(testo)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/sbin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)

if(GPU)
	add_definitions(-DGPU)
	enable_language("CUDA")
endif()

add_definitions(-DASIO_STANDALONE)

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/out/include
)

link_directories(
	${CMAKE_BINARY_DIR}/out/lib
)

macro(add_package NAME)
	# file name
	if ($ENV{BUILDKITE_BUILD_NUMBER})
		set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-$ENV{BUILDKITE_BUILD_NUMBER}")
		set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}-$ENV{BUILDKITE_BUILD_NUMBER}")
	else()
		set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
	endif()
	message(${CPACK_PACKAGE_FILE_NAME})

	# contact
	set(CPACK_PACKAGE_CONTACT "testo@testo.ru")

	set(CPACK_OUTPUT_CONFIG_FILE CPackConfig.cmake)
	set(CPACK_SOURCE_OUTPUT_CONFIG_FILE CPackSourceConfig.cmake)
	set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_CURRENT_BINARY_DIR};${NAME};ALL;/")
	include(CPack)

	add_custom_target(${NAME} cpack --config CPackConfig.cmake)
endmacro(add_package)

if(WIN32)
	set(EXTRA_FLAGS "/W0 -DNOMINMAX")
else()
	if(CMAKE_BUILD_TYPE MATCHES Release)
		set(EXTRA_FLAGS "${EXTRA_FLAGS} -Wall -Wextra -Wpedantic -O3 -DNDEBUG -g")
	elseif(CMAKE_BUILD_TYPE MATCHES Debug)
		set(EXTRA_FLAGS "${EXTRA_FLAGS} -O0 -D_DEBUG -g")
	else()
		MESSAGE(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE value: ${CMAKE_BUILD_TYPE}")
	endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(3rd_party)
add_subdirectory(negotiator)
add_subdirectory(tests)
add_subdirectory(src)
