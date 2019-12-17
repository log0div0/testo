cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "testo-guest-additions")
set(CPACK_PACKAGE_VERSION "1.0.0")

install(FILES
	${CMAKE_BINARY_DIR}/out/sbin/testo-guest-additions
	DESTINATION /usr/sbin
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/lib/systemd/system/testo-guest-additions.service
	DESTINATION /lib/systemd/system/
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST
	/usr
	/usr/sbin
	/lib
	/lib/systemd
	/lib/systemd/system
)

set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/postinst)
set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/prerm)
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/postrm)

set(CPACK_GENERATOR "RPM")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/out/pkg/")
add_package(guest-additions-package-rpm)

add_dependencies(guest-additions-package-rpm testo-guest-additions)
