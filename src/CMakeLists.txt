cmake_minimum_required(VERSION 3.0)

if (WIN32)
	add_definitions(-D__PRETTY_FUNCTION__=__FUNCSIG__ -DCOBJMACROS -DCINTERFACE -D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DVBOX_WITH_GLUE)
endif()

add_definitions(-DFMT_HEADER_ONLY)

include_directories(
	${CMAKE_SOURCE_DIR}/3rd_party
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_BINARY_DIR}/out/include
)

link_directories(
	${CMAKE_BINARY_DIR}/out/lib
)

add_subdirectory(vbox)
add_subdirectory(testo)
add_subdirectory(testow)
add_subdirectory(qemu)
