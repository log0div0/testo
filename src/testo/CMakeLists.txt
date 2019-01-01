cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)
add_executable(testo ${SRCS})

target_link_libraries(testo vbox vboxcapi)
add_dependencies(testo json)
if(UNIX)
	target_link_libraries(testo dl stdc++fs lept tesseract)
endif()
