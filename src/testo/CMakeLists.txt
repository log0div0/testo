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

add_executable(testo ${SRCS})
target_link_libraries(testo coro pugixml stb os guest_additions_common_stuff)

if(WIN32)
	target_link_libraries(testo hyperv msft virtdisk++)
endif()

if (UNIX AND NOT APPLE)
	target_link_libraries(testo qemu guestfs++)
endif()

if(UNIX)
	target_link_libraries(testo pthread)
endif()
