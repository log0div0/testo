cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)

#[[set(LIBS
	breakpad
	breakpad_client
	pthread
)]]

add_executable(guest_additions ${SRCS})

add_custom_target(guest_additions-iso
	COMMAND
		rm -f guest_additions.iso &&
		genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot -o guest_additions.iso ${CMAKE_BINARY_DIR}/out/sbin/guest_additions
	DEPENDS
		guest_additions
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/out/pkg)

#target_link_libraries(nf2-starter ${LIBS})

#add_dependencies(nf2-starter argagg libbreakpad spdlog)
