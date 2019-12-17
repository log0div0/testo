cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "testo-guest-additions")
set(CPACK_PACKAGE_VERSION "1.0.0")

install(FILES
	${CMAKE_BINARY_DIR}/out/sbin/testo-guest-additions
	DESTINATION /usr/sbin
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/etc/init.d/testo-guest-additions
	DESTINATION /etc/init.d
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
"${CMAKE_CURRENT_SOURCE_DIR}/postinst;\
${CMAKE_CURRENT_SOURCE_DIR}/prerm;\
${CMAKE_CURRENT_SOURCE_DIR}/postrm;")

set(CPACK_GENERATOR "DEB")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/out/pkg/")
add_package(guest-additions-package-deb)

add_dependencies(guest-additions-package-deb testo-guest-additions)
