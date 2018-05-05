cmake_minimum_required(VERSION 3.0)

add_definitions(-std=gnu++14)

file(GLOB_RECURSE SRCS *.cpp)

if (WIN32)
	list(APPEND SRCS ${CMAKE_SOURCE_DIR}/3rd_party/VBoxCAPI/VirtualBox_i_mscom.c)
else()
	list(APPEND SRCS ${CMAKE_SOURCE_DIR}/3rd_party/VBoxCAPI/VirtualBox_i_xpcom.c)
endif()

add_executable(testo ${SRCS})

if (UNIX)
	target_link_libraries(testo dl pthread)
endif()
