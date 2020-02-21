cmake_minimum_required(VERSION 3.15)

cmake_policy(SET CMP0091 NEW)

project(testo)

set(TESTO_VERSION "1.3.0")

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/sbin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)

add_definitions(-DASIO_STANDALONE)

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party
	${CMAKE_SOURCE_DIR}/src
)

include_directories(
	${CMAKE_BINARY_DIR}/out/include
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
	set(CPACK_PACKAGE_CONTACT "s.lvov@testo-lang.ru")

	set(CPACK_OUTPUT_CONFIG_FILE CPackConfig.cmake)
	set(CPACK_SOURCE_OUTPUT_CONFIG_FILE CPackSourceConfig.cmake)
	set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_CURRENT_BINARY_DIR};${NAME};ALL;/")
	include(CPack)

	add_custom_target(${NAME} cpack --config CPackConfig.cmake)
endmacro(add_package)

if(APPLE)
	add_definitions(-D_XOPEN_SOURCE)
endif()

if(WIN32)
	add_definitions(
		-D_WIN32_WINNT=0x0601
		-D_SCL_SECURE_NO_WARNINGS
		-D_WINSOCK_DEPRECATED_NO_WARNINGS
		-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
		-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
		-DNOMINMAX
	)
else()
	if(CMAKE_BUILD_TYPE MATCHES Release)
		set(EXTRA_FLAGS "-Wall -Wextra -Wpedantic -Wno-unused-parameter -O2 -DNDEBUG -g")
	elseif(CMAKE_BUILD_TYPE MATCHES Debug)
		set(EXTRA_FLAGS "-O0 -D_DEBUG -g")
	else()
		MESSAGE(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE value: ${CMAKE_BUILD_TYPE}")
	endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(3rd_party)
if (NOT APPLE)
	add_subdirectory(guest_additions)
endif()
if (UNIX AND NOT APPLE)
	add_subdirectory(pkg)
	add_subdirectory(docs)
	add_subdirectory(tests)
endif()
add_subdirectory(src)
