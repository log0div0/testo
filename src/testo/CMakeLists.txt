cmake_minimum_required(VERSION 3.0)

add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")
add_definitions(-DTESTO_RELEASE_DATE="${TESTO_RELEASE_DATE}")

list(APPEND SRC_DIRS "*.cpp")

list(APPEND SRC_DIRS "backends/*.cpp")
if (WIN32)
list(APPEND SRC_DIRS "backends/hyperv/*.cpp")
endif()
if (UNIX AND NOT APPLE)
list(APPEND SRC_DIRS "backends/qemu/*.cpp")
endif()

list(APPEND SRC_DIRS "IR/*.cpp")

file(GLOB SRCS ${SRC_DIRS})
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_library(testo_core ${SRCS})
target_link_libraries(testo_core coro pugixml stb os testo_guest_additions_protocol)

if(WIN32)
	target_link_libraries(testo_core hyperv msft virtdisk++)
endif()

if (UNIX AND NOT APPLE)
	target_link_libraries(testo_core qemu guestfs++)
endif()

if(UNIX)
	target_link_libraries(testo_core pthread)
endif()

add_executable(testo ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(testo testo_core)

macro(use_breakpad NAME)
	target_compile_options(${NAME} PUBLIC -g)
	target_compile_definitions(${NAME} PUBLIC -DUSE_BREAKPAD=1)
	add_custom_target(${NAME}-without-symbols
		COMMAND strip --strip-debug ${CMAKE_BINARY_DIR}/out/sbin/${NAME}
			-o ${CMAKE_BINARY_DIR}/out/sbin/${NAME}-without-symbols
		COMMAND mkdir -p ${CMAKE_BINARY_DIR}/out/symbols/
		COMMAND ${BREAKPAD_DIR}/bin/dump_syms ${CMAKE_BINARY_DIR}/out/sbin/${NAME} > ${CMAKE_BINARY_DIR}/out/symbols/${NAME}.sym
		COMMAND rm -rf ${CMAKE_BINARY_DIR}/out/symbols/${NAME}
		COMMAND mkdir -p ${CMAKE_BINARY_DIR}/out/symbols/${NAME}/`head -n1 ${CMAKE_BINARY_DIR}/out/symbols/${NAME}.sym | cut -d\  -f4`
		COMMAND mv ${CMAKE_BINARY_DIR}/out/symbols/${NAME}.sym ${CMAKE_BINARY_DIR}/out/symbols/${NAME}/`head -n1 ${CMAKE_BINARY_DIR}/out/symbols/${NAME}.sym | cut -d\  -f4`
		DEPENDS ${NAME}
	)
	target_include_directories(${NAME} PUBLIC ${BREAKPAD_DIR}/include/breakpad)
	target_link_directories(${NAME} PUBLIC ${BREAKPAD_DIR}/lib)
	target_link_libraries(${NAME} breakpad_client)
endmacro(use_breakpad)

if(BREAKPAD_DIR)
	use_breakpad(testo)
endif()

file(GLOB UNIT_TESTS_SRCS "unit_tests/*.cpp")
add_executable(testo_unit_tests ${UNIT_TESTS_SRCS})
target_link_libraries(testo_unit_tests testo_core)

add_subdirectory_if_exists(pkg)
