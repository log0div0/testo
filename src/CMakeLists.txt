cmake_minimum_required(VERSION 3.0)

add_definitions(-DFMT_HEADER_ONLY)

if (WIN32)
	add_definitions(-D__PRETTY_FUNCTION__=__FUNCSIG__)
endif()

if (APPLE)
	include_directories(/usr/local/include)
	link_directories(/usr/local/lib)
endif()

add_definitions(-DNN_SERVICE_PROCOTOL_VERSION="1.0")

add_subdirectory_if_exists(testo)

if (TESTO_GUEST_ADDITIONS_HYPERVISOR)
add_subdirectory_if_exists(testo_guest_additions)
endif()

add_subdirectory_if_exists(testo_guest_additions_protocol)
add_subdirectory_if_exists(testo_nn_service)
