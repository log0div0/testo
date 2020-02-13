cmake_minimum_required(VERSION 3.0)

add_subdirectory(src)
add_subdirectory(pkg)

if (UNIX)
	add_custom_target(testo-guest-additions-iso
		COMMAND
			rm -f testo-guest-additions.iso &&
			genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot -o testo-guest-additions.iso ${CMAKE_BINARY_DIR}/out/pkg/testo-guest-additions-1.0.0.deb ${CMAKE_BINARY_DIR}/out/pkg/testo-guest-additions-1.0.0.rpm
		DEPENDS
			testo-guest-additions-package
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/out/pkg)
endif()
