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
list(APPEND SRC_DIRS "js/*.cpp")

file(GLOB SRCS ${SRC_DIRS})
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_library(testo_core ${SRCS})
target_link_libraries(testo_core coro pugixml stb os testo_guest_additions_protocol quickjs)

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

file(GLOB UNIT_TESTS_SRCS "unit_tests/*.cpp")
add_executable(testo_unit_tests ${UNIT_TESTS_SRCS})
target_link_libraries(testo_unit_tests testo_core)

add_subdirectory_if_exists(pkg)
