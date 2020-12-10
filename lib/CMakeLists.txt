cmake_minimum_required(VERSION 3.0)

add_subdirectory(process)

if (WIN32)
add_subdirectory(virtdisk)
add_subdirectory(msft)
add_subdirectory(hyperv)
endif()

if (UNIX AND NOT APPLE)
add_subdirectory(qemu)
add_subdirectory(guestfs)
endif()

add_subdirectory(stb)
add_subdirectory(psf)
