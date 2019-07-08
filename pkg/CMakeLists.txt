cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "testo")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/out/pkg/")
set(CPACK_GENERATOR "DEB")

macro(install_without_symbols NAME)
	add_custom_target(${NAME}-without-symbols
		COMMAND strip --strip-debug ${CMAKE_BINARY_DIR}/out/sbin/${NAME}
			-o ${CMAKE_BINARY_DIR}/out/sbin/${NAME}-without-symbols
		DEPENDS ${NAME}
	)
	install(PROGRAMS
		${CMAKE_BINARY_DIR}/out/sbin/${NAME}-without-symbols
		DESTINATION /usr/sbin/
		RENAME ${NAME}
		PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
	)
endmacro(install_without_symbols)

install_without_symbols(testo)

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
	"${CMAKE_CURRENT_SOURCE_DIR}/preinst;\
	${CMAKE_CURRENT_SOURCE_DIR}/postrm;")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "\
	libvirt0 (>= 4.0.0), \
	libvirt-bin (>= 4.0.0), \
	qemu (>= 1.2.11), \
	qemu-kvm (>= 2.11), \
	libopenblas-base (>= 0.2.0) \
")

add_package(testo-package)
add_dependencies(testo-package testo-without-symbols)
