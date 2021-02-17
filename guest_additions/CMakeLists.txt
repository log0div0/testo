cmake_minimum_required(VERSION 3.0)

string(TOUPPER ${TESTO_GUEST_ADDITIONS_HYPERVISOR} TESTO_GUEST_ADDITIONS_HYPERVISOR)

add_subdirectory(src)
add_subdirectory(pkg)
