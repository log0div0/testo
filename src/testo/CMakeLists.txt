cmake_minimum_required(VERSION 3.0)

list(APPEND SRC_DIRS "*.cpp")

list(APPEND SRC_DIRS "backends/*.cpp")
list(APPEND SRC_DIRS "backends/dummy/*.cpp")
list(APPEND SRC_DIRS "backends/vbox/*.cpp")
if (WIN32)
list(APPEND SRC_DIRS "backends/hyperv/*.cpp")
endif()
if (UNIX AND NOT APPLE)
list(APPEND SRC_DIRS "backends/qemu/*.cpp")
endif()

list(APPEND SRC_DIRS "screen_selection/*.cpp")

file(GLOB SRCS ${SRC_DIRS})
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_library(testo-core STATIC ${SRCS})
target_link_libraries(testo-core coro vbox pugixml stb text-detector)

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
