cmake_minimum_required(VERSION 3.0)

if (WIN32)
add_subdirectory(virtdisk)
add_subdirectory(msft)
add_subdirectory(wmi)
add_subdirectory(hyperv)
add_subdirectory(winapi)
endif()

if (UNIX)
add_subdirectory(posixapi)
endif()

if (UNIX AND NOT APPLE)
add_subdirectory(qemu)
add_subdirectory(guestfs)
endif()

add_subdirectory(stb)
add_subdirectory(psf)
add_subdirectory(os)
