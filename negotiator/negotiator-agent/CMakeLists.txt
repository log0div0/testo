cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "negotiator-agent")
set(CPACK_PACKAGE_VERSION "0.0.1")

install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/usr/share/pyshared/negotiator-agent.py
	${CMAKE_SOURCE_DIR}/negotiator/negotiator.py
	DESTINATION /usr/share/pyshared)

install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/usr/sbin/negotiator-agent
	DESTINATION /usr/sbin
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/etc/init.d/negotiator-agent
	DESTINATION /etc/init.d
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
"${CMAKE_CURRENT_SOURCE_DIR}/postinst;\
${CMAKE_CURRENT_SOURCE_DIR}/prerm;\
${CMAKE_CURRENT_SOURCE_DIR}/postrm;")

set(CPACK_GENERATOR "DEB")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/out/pkg/")
add_package(negotiator-agent-package)

add_custom_target(negotiator-iso
	COMMAND
		rm -f negotiator.iso &&
		genisoimage -f -J -joliet-long -r -allow-lowercase -allow-multidot -o negotiator.iso negotiator-agent-0.0.1.deb
	DEPENDS
		negotiator-agent-package
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/out/pkg)
