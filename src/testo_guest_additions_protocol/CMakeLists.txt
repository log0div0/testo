cmake_minimum_required(VERSION 3.0)

add_definitions(-DTESTO_VERSION="${TESTO_VERSION}")

if (WIN32)
	add_definitions(-D__PRETTY_FUNCTION__=__FUNCSIG__)
endif()

file(GLOB_RECURSE SRCS *.cpp)
add_library(testo_guest_additions_protocol STATIC ${SRCS})
target_link_libraries(testo_guest_additions_protocol os)