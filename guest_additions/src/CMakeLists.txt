cmake_minimum_required(VERSION 3.0)

add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")

set(SRCS Server.cpp)

if (UNIX AND NOT APPLE)
	list(APPEND SRCS MainLinux.cpp)
	set(CMAKE_EXE_LINKER_FLAGS "-static")
endif()
if (WIN32)
	list(APPEND SRCS MainWin.cpp)
endif()

if (TESTO_GUEST_ADDITIONS_HYPERVISOR STREQUAL "QEMU")
	if (UNIX AND NOT APPLE)
		set(CHANNEL_CLASS_NAME "QemuLinuxChannel")
		list(APPEND SRCS QemuLinuxChannel.cpp)
	endif()
	if (WIN32)
		set(CHANNEL_CLASS_NAME "QemuWinChannel")
		list(APPEND SRCS QemuWinChannel.cpp QemuWinChannelExtra.cpp)
	endif()
elseif(TESTO_GUEST_ADDITIONS_HYPERVISOR STREQUAL "HYPERV")
	set(CHANNEL_CLASS_NAME "HyperVChannel")
	list(APPEND SRCS HyperVChannel.cpp)
else()
	message(FATAL_ERROR "Unknown TESTO_GUEST_ADDITIONS_HYPERVISOR: ${TESTO_GUEST_ADDITIONS_HYPERVISOR}")
endif()

add_definitions(-DCHANNEL_CLASS_NAME=${CHANNEL_CLASS_NAME})
add_definitions(-DCHANNEL_CLASS_HEADER="${CHANNEL_CLASS_NAME}.hpp")

add_executable(testo-guest-additions ${SRCS})
target_link_libraries(testo-guest-additions os coro)

if (UNIX)
	target_link_libraries(testo-guest-additions pthread)
endif()

if (WIN32)
	target_link_libraries(testo-guest-additions SetupAPI)
endif()
