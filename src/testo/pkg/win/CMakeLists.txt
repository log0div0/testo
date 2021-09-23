cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "Testo")
set(CPACK_PACKAGE_VERSION "${TESTO_VERSION}")
set(CPACK_WIX_UPGRADE_GUID "D80951FA-43DF-4B00-87CC-43A2824620D6")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Testo Lang/Testo")

install(TARGETS testo)

set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patch.xml")

add_package(testo-package)
add_dependencies(testo-package testo)
