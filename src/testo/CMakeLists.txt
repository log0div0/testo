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

file(GLOB NN_SRCS nn/*.cpp)
set(SRCS ${SRCS} ${NN_SRCS})

add_library(testo-core STATIC ${SRCS})
target_link_libraries(testo-core coro vbox pugixml)

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
