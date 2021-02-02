cmake_minimum_required(VERSION 3.0)

add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")

if (TESTO_GUEST_ADDITIONS_HYPERVISOR STREQUAL "QEMU")
	set(HYPERVISOR_NAME "Qemu")
elseif(TESTO_GUEST_ADDITIONS_HYPERVISOR STREQUAL "HYPERV")
	set(HYPERVISOR_NAME "HyperV")
else()
	message(FATAL_ERROR "Unknown TESTO_GUEST_ADDITIONS_HYPERVISOR: ${TESTO_GUEST_ADDITIONS_HYPERVISOR}")
endif()

if (UNIX AND NOT APPLE)
	set(CMAKE_EXE_LINKER_FLAGS "-static")
	set(OS_NAME "Linux")
endif()

if (WIN32)
	set(OS_NAME "Win")
endif()

add_definitions(-DCHANNEL_CLASS_HEADER="${HYPERVISOR_NAME}${OS_NAME}Channel.hpp")
add_definitions(-DCHANNEL_CLASS_NAME=${HYPERVISOR_NAME}${OS_NAME}Channel)

set(SRCS
	Server.cpp
	${HYPERVISOR_NAME}${OS_NAME}Channel.cpp
	Main${OS_NAME}.cpp
)

if ((HYPERVISOR_NAME STREQUAL "Qemu") AND (OS_NAME STREQUAL "Win"))
	list(APPEND SRCS QemuWinChannelExtra.cpp)
endif()

add_executable(testo-guest-additions ${SRCS})

if (UNIX)
	target_link_libraries(testo-guest-additions os pthread)
endif()

if (WIN32)
	target_link_libraries(testo-guest-additions SetupAPI os coro)
endif()
