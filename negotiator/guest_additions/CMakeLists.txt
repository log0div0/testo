cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE SRCS *.cpp)

set(LIBS
	coro
	pthread
)

add_executable(guest_additions ${SRCS})
target_link_libraries(guest_additions ${LIBS})

add_custom_target(guest_additions-iso
	COMMAND
		rm -f guest_additions.iso &&
		genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot -o guest_additions.iso ${CMAKE_BINARY_DIR}/out/sbin/guest_additions
	DEPENDS
		guest_additions
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/out/pkg)


#add_dependencies(nf2-starter argagg libbreakpad spdlog)
