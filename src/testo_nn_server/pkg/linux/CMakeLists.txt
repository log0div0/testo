cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "testo-nn-server")
set(CPACK_PACKAGE_VERSION "${TESTO_VERSION}")

install(PROGRAMS
	${CMAKE_CURRENT_SOURCE_DIR}/usr/sbin/testo-nn-server
	${CMAKE_CURRENT_SOURCE_DIR}/usr/sbin/testo-request-license
	DESTINATION /usr/sbin
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(TARGETS testo_nn_server
	DESTINATION /usr/libexec
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(TARGETS testo_request_license
	DESTINATION /usr/libexec
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/init.d
	DESTINATION /etc/init.d
	RENAME testo-nn-server
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(FILES
	${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so.1.5.3
	DESTINATION /usr/lib/testo
	PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

install(FILES
	${CMAKE_SOURCE_DIR}/src/testo_nn_server/nn/TextColorPicker.onnx
	${CMAKE_SOURCE_DIR}/src/testo_nn_server/nn/TextDetector.onnx
	${CMAKE_SOURCE_DIR}/src/testo_nn_server/nn/TextRecognizer.onnx
	DESTINATION /usr/share/testo
	PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST
	/usr
	/usr/sbin
	/usr/lib
	/usr/libexec
	/usr/share
	/etc
	/etc/init.d
)

set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/postinst)
set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/prerm)
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/postrm)

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
"${CMAKE_CURRENT_SOURCE_DIR}/postinst;\
${CMAKE_CURRENT_SOURCE_DIR}/prerm;\
${CMAKE_CURRENT_SOURCE_DIR}/postrm;")

set(CPACK_RPM_PACKAGE_PROVIDES "\
	libonnxruntime.so.1.5.3()(64bit), \
	libonnxruntime.so.1.5.3(VERS_1.5.3)(64bit) \
")

add_package(testo-nn-server-package)
add_dependencies(testo-nn-server-package testo_nn_server testo_request_license)
