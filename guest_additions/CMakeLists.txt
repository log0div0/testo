cmake_minimum_required(VERSION 3.0)

add_subdirectory(src)
add_subdirectory(pkg)

if (UNIX)
	set(ISO_FILES
		${CMAKE_CURRENT_SOURCE_DIR}/iso/testo-guest-additions.deb
		${CMAKE_CURRENT_SOURCE_DIR}/iso/testo-guest-additions.rpm
		${CMAKE_CURRENT_SOURCE_DIR}/iso/autorun.inf
		${CMAKE_CURRENT_SOURCE_DIR}/iso/autorun.js
		${CMAKE_CURRENT_SOURCE_DIR}/iso/testo-guest-additions-x86.msi
		${CMAKE_CURRENT_SOURCE_DIR}/iso/testo-guest-additions-x64.msi
	)

	add_custom_target(testo-guest-additions-iso
		COMMAND
			rm -f testo-guest-additions.iso &&
			genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot -o testo-guest-additions-${TESTO_VERSION}.iso ${ISO_FILES}
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/out/pkg)
endif()
