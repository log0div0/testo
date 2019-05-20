cmake_minimum_required(VERSION 3.0)

if (CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-D_DEBUG)
endif()

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

if (WIN32)
	list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/QemuFlashDriveController.cpp)
	list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/QemuGuestAdditions.cpp)
	list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/QemuVmController.cpp)
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
target_link_libraries(testo-core coro vbox vboxcapi pugixml libdarknet)

if(WIN32)
	target_link_libraries(testo-core hyperv)
endif()

if(UNIX)
	target_link_libraries(testo-core dl stdc++fs qemu pthread)
endif()

if(GPU)
	target_link_libraries(testo-core cuda cudart cublas curand)
endif()

add_executable(testo main.cpp)
add_dependencies(testo testo-core)
target_link_libraries(testo testo-core)
