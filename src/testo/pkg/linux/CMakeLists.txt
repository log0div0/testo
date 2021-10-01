cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "testo")
set(CPACK_PACKAGE_VERSION "${TESTO_VERSION}")

install(PROGRAMS
	${CMAKE_CURRENT_SOURCE_DIR}/usr/sbin/testo
	DESTINATION /usr/sbin
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if(BREAKPAD_DIR)
install(PROGRAMS ${CMAKE_BINARY_DIR}/out/sbin/testo-without-symbols
	DESTINATION /usr/libexec
	RENAME testo
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
else()
install(TARGETS testo
	DESTINATION /usr/libexec
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST
	/usr
	/usr/sbin
)

set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/preinst)
set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/postinst)
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/postrm)

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	"${CMAKE_CURRENT_SOURCE_DIR}/preinst;\
	${CMAKE_CURRENT_SOURCE_DIR}/postinst;\
	${CMAKE_CURRENT_SOURCE_DIR}/postrm;")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "\
	libvirt0 (>= 3.0.0), \
	libvirt-clients (>= 3.0.0), \
	libvirt-daemon-system (>= 3.0.0), \
	qemu (>= 2.8), \
	qemu-kvm, \
	ebtables, \
	dnsmasq-base, \
	libguestfs0 (>= 1.34.6) \
")

set(CPACK_RPM_PACKAGE_REQUIRES "\
	libvirt >= 3.0.0 \
	qemu-kvm \
	iptables-ebtables \
	dnsmasq \
	libguestfs >= 1.38.4 \
")

add_package(testo-package)
add_dependencies(testo-package testo)
if(BREAKPAD_DIR)
	add_dependencies(testo-package testo-without-symbols)
endif()
