cmake_minimum_required(VERSION 3.0)

if (WIN32)
	add_definitions(-D__PRETTY_FUNCTION__=__FUNCSIG__ -DCOBJMACROS -DCINTERFACE -D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DVBOX_WITH_GLUE)
endif()

add_definitions(-DFMT_HEADER_ONLY)

add_subdirectory(vbox)
if (UNIX AND NOT APPLE)
add_subdirectory(qemu)
add_subdirectory(testo)
add_subdirectory(testow)
endif()
