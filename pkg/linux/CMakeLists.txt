cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "testo")
set(CPACK_PACKAGE_VERSION "${TESTO_VERSION}")

install(PROGRAMS
	${CMAKE_BINARY_DIR}/out/sbin/testo
	DESTINATION /usr/sbin
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(FILES
	${ONNX_RUNTIME_DIR}/lib/libonnxruntime_testo.so.1.3.0
	DESTINATION /usr/lib
	PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

install(FILES
	${CMAKE_SOURCE_DIR}/src/nn/TextDetector.onnx
	${CMAKE_SOURCE_DIR}/src/nn/TextRecognizer.onnx
	DESTINATION /usr/share/testo
	PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

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
	qemu-kvm (>= 2.8), \
	ebtables, \
	dnsmasq-base, \
	libguestfs0 (>= 1.34.6) \
")

set(CPACK_RPM_PACKAGE_PROVIDES "\
	libonnxruntime_testo.so.1.3.0()(64bit), \
	libonnxruntime_testo.so.1.3.0(VERS_1.3.0)(64bit) \
")

set(CPACK_RPM_PACKAGE_REQUIRES "\
	libvirt >= 3.0.0 \
	qemu-kvm >= 2.8 \
	iptables-ebtables \
	dnsmasq \
	libguestfs >= 1.38.4 \
")

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST
	/usr
	/usr/sbin
	/usr/lib
	/usr/share/
)

add_package(testo-package)
add_dependencies(testo-package testo)
