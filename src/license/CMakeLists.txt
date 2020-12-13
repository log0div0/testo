cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(license STATIC ${SRCS})
target_link_libraries(license tweetnacl)
set(CMAKE_EXE_LINKER_FLAGS "-static")
add_executable(testo_license_cgi main.cpp)
target_link_libraries(testo_license_cgi tweetnacl license)
