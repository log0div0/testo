cmake_minimum_required(VERSION 3.0)

add_subdirectory(deb)
add_subdirectory(rpm)

add_custom_target(negotiator-iso
	COMMAND
		rm -f negotiator.iso &&
		genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot -o negotiator.iso negotiator-agent-0.0.1.deb negotiator-agent-0.0.1.rpm
	DEPENDS
		negotiator-agent-package-deb negotiator-agent-package-rpm
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/out/pkg)
