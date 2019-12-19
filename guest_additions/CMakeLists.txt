cmake_minimum_required(VERSION 3.0)

add_subdirectory(deb)
add_subdirectory(rpm)

file(GLOB_RECURSE SRCS *.cpp)

set(LIBS
	stdc++fs
	)

set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(testo-guest-additions ${SRCS})
target_link_libraries(testo-guest-additions ${LIBS})

add_custom_target(guest-additions-iso
	COMMAND
		rm -f testo-guest-additions.iso &&
		genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot -o testo-guest-additions.iso ${CMAKE_BINARY_DIR}/out/pkg/testo-guest-additions-1.0.0.deb ${CMAKE_BINARY_DIR}/out/pkg/testo-guest-additions-1.0.0.rpm
	DEPENDS
		guest-additions-package-deb guest-additions-package-rpm
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/out/pkg)
