cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "negotiator-agent")
set(CPACK_PACKAGE_VERSION "0.0.1")

install(FILES
	${CMAKE_SOURCE_DIR}/negotiator/negotiator-agent/usr/share/pyshared/negotiator-agent.py
	${CMAKE_SOURCE_DIR}/negotiator/negotiator.py
	DESTINATION /usr/share/pyshared)

install(FILES
	${CMAKE_SOURCE_DIR}/negotiator/negotiator-agent/usr/sbin/negotiator-agent
	DESTINATION /usr/sbin
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/lib/systemd/system/negotiator-agent.service
	DESTINATION /lib/systemd/system/
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST
	/usr
	/usr/sbin
	/usr/share
	/usr/share/pyshared
	/lib
	/lib/systemd
	/lib/systemd/system
)

set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/postinst)
set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/prerm)
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/postrm)

set(CPACK_GENERATOR "RPM")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/out/pkg/")
add_package(negotiator-agent-package-rpm)