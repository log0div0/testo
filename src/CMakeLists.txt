cmake_minimum_required(VERSION 3.0)

if (WIN32)
	add_definitions(-D__PRETTY_FUNCTION__=__FUNCSIG__ -DCOBJMACROS -DCINTERFACE -D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DVBOX_WITH_GLUE)
endif()

include_directories(
	${CMAKE_BINARY_DIR}/out/include
	${CMAKE_SOURCE_DIR}/include
	${CMAKE_SOURCE_DIR}/3rd_party
)

file(GLOB_RECURSE SRCS *.cpp)
link_directories(${CMAKE_BINARY_DIR}/out/lib)
add_executable(testo ${SRCS})

target_link_libraries(testo vboxcapi)
add_dependencies(testo json)
if(UNIX)
	target_link_libraries(testo dl stdc++fs lept tesseract)
	add_definitions(-std=c++17)
endif()
