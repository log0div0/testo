cmake_minimum_required(VERSION 3.0)

add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")

set(SRCS MessageHandler.cpp)

if (UNIX AND NOT APPLE)
	list(APPEND SRCS MainLinux.cpp)
	set(CMAKE_EXE_LINKER_FLAGS "-static")
endif()
if (WIN32)
	list(APPEND SRCS MainWin.cpp)
endif()

if (TESTO_GUEST_ADDITIONS_HYPERVISOR STREQUAL "QEMU")
	add_definitions(-D__QEMU__)
	if (UNIX AND NOT APPLE)
		list(APPEND SRCS QemuLinuxChannel.cpp)
	endif()
	if (WIN32)
		list(APPEND SRCS QemuWinChannel.cpp QemuWinChannelExtra.cpp)
	endif()
elseif(TESTO_GUEST_ADDITIONS_HYPERVISOR STREQUAL "HYPERV")
	add_definitions(-D__HYPERV__)
	list(APPEND SRCS HyperVChannel.cpp)
else()
	message(FATAL_ERROR "Unknown TESTO_GUEST_ADDITIONS_HYPERVISOR: ${TESTO_GUEST_ADDITIONS_HYPERVISOR}")
endif()

add_executable(testo-guest-additions ${SRCS})
target_link_libraries(testo-guest-additions os coro)

add_executable(testo-guest-additions-cli CLI.cpp)
target_link_libraries(testo-guest-additions-cli os coro)

if (UNIX)
	target_link_libraries(testo-guest-additions pthread)
	target_link_libraries(testo-guest-additions-cli pthread)
endif()

if (WIN32)
	target_link_libraries(testo-guest-additions SetupAPI)
endif()
