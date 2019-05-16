cmake_minimum_required(VERSION 3.0)

add_definitions(-DFMT_HEADER_ONLY)

add_subdirectory(darknet)
add_subdirectory(vbox)
if (WIN32)
add_subdirectory(hyperv)
endif()
if (UNIX AND NOT APPLE)
add_subdirectory(qemu)
endif()
add_subdirectory(testo)
add_subdirectory(testow)
