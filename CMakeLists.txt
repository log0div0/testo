cmake_minimum_required(VERSION 3.15)

cmake_policy(SET CMP0091 NEW)

project(testo)

set(TESTO_VERSION "2.2.9")
set(TESTO_RELEASE_DATE "2020.12.19")

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
if (NOT CPACK_GENERATOR)
	set(CPACK_GENERATOR "DEB")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/sbin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out/lib)
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/out/pkg/")

add_definitions(-DASIO_STANDALONE -DASIO_ENABLE_CANCELIO)

if(WIN32)
	add_definitions(
		-D_WIN32_WINNT=0x0601
		-D_SCL_SECURE_NO_WARNINGS
		-D_WINSOCK_DEPRECATED_NO_WARNINGS
		-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
		-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
		-DNOMINMAX
		-DUNICODE
		-D_UNICODE
	)
endif()

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party
)

add_subdirectory(3rd_party)

macro(add_package NAME)
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}")
	set(CPACK_PACKAGE_CONTACT "support@testo-lang.ru")
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
	add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
	add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W1 /WX")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1 /WX")
else()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter")
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(
	${CMAKE_SOURCE_DIR}/lib
	${CMAKE_SOURCE_DIR}/src
)

macro(add_subdirectory_if_exists NAME)
	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}")
		add_subdirectory(${NAME})
	endif()
endmacro()

add_subdirectory_if_exists(lib)
add_subdirectory_if_exists(src)
add_subdirectory_if_exists(nn)
if (NOT APPLE)
	add_subdirectory_if_exists(guest_additions)
	add_subdirectory_if_exists(pkg)
endif()
