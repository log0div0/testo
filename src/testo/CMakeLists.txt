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

list(APPEND SRC_DIRS "tql/*.cpp")

file(GLOB SRCS ${SRC_DIRS})

add_executable(testo ${SRCS})

set(CMAKE_EXE_LINKER_FLAGS "${ONNX_RUNTIME_LINKER_FLAGS}")

target_link_libraries(testo coro quickjs-c++ vbox pugixml stb nn license)

if(WIN32)
	target_link_libraries(testo hyperv msft virtdisk++)
endif()

if (UNIX AND NOT APPLE)
	target_link_libraries(testo qemu)
endif()

if(UNIX)
	target_link_libraries(testo stdc++fs pthread)
endif()
