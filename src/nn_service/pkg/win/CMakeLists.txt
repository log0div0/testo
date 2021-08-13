cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "Testo NN Service")
set(CPACK_PACKAGE_VERSION "${TESTO_VERSION}")
set(CPACK_WIX_UPGRADE_GUID "41DBA69D-9F7A-4BE2-8635-08C84492CA1D")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Testo Lang/Testo NN Serice")

file(TO_CMAKE_PATH "${ONNX_RUNTIME_DIR}/bin/onnxruntime.dll" ONNX_RUNTIME_DLL)
install(PROGRAMS ${ONNX_RUNTIME_DLL} DESTINATION bin)

install(FILES
	${CMAKE_SOURCE_DIR}/src/nn/Homm3Detector.onnx
	${CMAKE_SOURCE_DIR}/src/nn/TextColorPicker.onnx
	${CMAKE_SOURCE_DIR}/src/nn/TextDetector.onnx
	${CMAKE_SOURCE_DIR}/src/nn/TextRecognizer.onnx
	DESTINATION share)

install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/nn_service.json
	DESTINATION .)

include(InstallRequiredSystemLibraries)

list(APPEND CPACK_WIX_CANDLE_EXTRA_FLAGS "-dCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
list(APPEND CPACK_WIX_CANDLE_EXTRA_FLAGS "-dCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

set(CPACK_WIX_EXTENSIONS "WixDifxAppExtension")
set(CPACK_WIX_CUSTOM_XMLNS "difx=http://schemas.microsoft.com/wix/DifxAppExtension")

set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patch.xml")

add_package(testo-nn-service-package)
add_dependencies(testo-nn-service-package testo_nn_service)
