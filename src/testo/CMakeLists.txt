cmake_minimum_required(VERSION 3.0)

if (CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-D_DEBUG)
endif()

file(GLOB SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

file(GLOB BACKENDS_SRCS backends/*.cpp)
set(SRCS ${SRCS} ${BACKENDS_SRCS})

file(GLOB DUMMY_SRCS backends/dummy/*.cpp)
set(SRCS ${SRCS} ${DUMMY_SRCS})

file(GLOB VBOX_SRCS backends/vbox/*.cpp)
set(SRCS ${SRCS} ${VBOX_SRCS})

if (WIN32)
	file(GLOB HYPERV_SRCS backends/hyperv/*.cpp)
	set(SRCS ${SRCS} ${HYPERV_SRCS})
endif()
if (UNIX AND NOT APPLE)
	file(GLOB QEMU_SRCS backends/qemu/*.cpp)
	set(SRCS ${SRCS} ${QEMU_SRCS})
endif()

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/testo_network.cpp
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/nn/
	COMMAND xxd -i testo.network ${CMAKE_BINARY_DIR}/src/testo_network.cpp
	DEPENDS ${CMAKE_SOURCE_DIR}/nn/testo.network
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/testo_weights.cpp
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/nn/
	COMMAND xxd -i testo.weights ${CMAKE_BINARY_DIR}/src/testo_weights.cpp
	DEPENDS ${CMAKE_SOURCE_DIR}/nn/testo.weights
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/testo_symbols.cpp
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/nn/
	COMMAND xxd -i testo.symbols ${CMAKE_BINARY_DIR}/src/testo_symbols.cpp
	DEPENDS ${CMAKE_SOURCE_DIR}/nn/testo.symbols
)

list(APPEND SRCS ${CMAKE_BINARY_DIR}/src/testo_network.cpp)
list(APPEND SRCS ${CMAKE_BINARY_DIR}/src/testo_weights.cpp)
list(APPEND SRCS ${CMAKE_BINARY_DIR}/src/testo_symbols.cpp)


add_library(testo-core STATIC ${SRCS})
add_dependencies(testo-core darknet)
target_link_libraries(testo-core coro vbox pugixml libdarknet)

if(WIN32)
	target_link_libraries(testo-core hyperv msft virtdisk++)
endif()

if (UNIX AND NOT APPLE)
	target_link_libraries(testo-core qemu)
endif()

if(UNIX)
	target_link_libraries(testo-core stdc++fs pthread)
endif()

add_executable(testo main.cpp)
add_dependencies(testo testo-core)
target_link_libraries(testo testo-core)
