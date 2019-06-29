cmake_minimum_required(VERSION 3.0)

add_definitions(-DFMT_HEADER_ONLY)
if (WIN32)
	add_definitions(
		-D__PRETTY_FUNCTION__=__FUNCSIG__
		/W1
		/WX
	)
endif()

add_subdirectory(darknet)
add_subdirectory(vbox)
add_subdirectory(opencl++)
if (WIN32)
add_subdirectory(virtdisk)
add_subdirectory(msft)
add_subdirectory(hyperv)
endif()
if (UNIX AND NOT APPLE)
add_subdirectory(qemu)
endif()
add_subdirectory(testo)
add_subdirectory(testow)
