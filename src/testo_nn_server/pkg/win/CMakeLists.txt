cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "Testo NN Server")
set(CPACK_PACKAGE_VERSION "${TESTO_VERSION}")
set(CPACK_WIX_UPGRADE_GUID "41DBA69D-9F7A-4BE2-8635-08C84492CA1D")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Testo Lang/Testo NN Server")

file(TO_CMAKE_PATH "${ONNX_RUNTIME_DIR}/bin/onnxruntime.dll" ONNX_RUNTIME_DLL)
install(PROGRAMS ${ONNX_RUNTIME_DLL} DESTINATION bin)

install(FILES
	${CMAKE_SOURCE_DIR}/src/testo_nn_server/nn/TextColorPicker.onnx
	${CMAKE_SOURCE_DIR}/src/testo_nn_server/nn/TextDetector.onnx
	${CMAKE_SOURCE_DIR}/src/testo_nn_server/nn/TextRecognizer.onnx
	DESTINATION share)

include(InstallRequiredSystemLibraries)

if (USE_CUDA)
	find_package(CUDAToolkit)
	file(TO_CMAKE_PATH "${CUDAToolkit_BIN_DIR}/cublas64_11.dll" CUBLAS_DLL)
	file(TO_CMAKE_PATH "${CUDAToolkit_BIN_DIR}/cublasLt64_11.dll" CUBLASLT_DLL)
	file(TO_CMAKE_PATH "${CUDAToolkit_BIN_DIR}/curand64_10.dll" CURAND_DLL)
	install(PROGRAMS
		${CUBLAS_DLL}
		${CUBLASLT_DLL}
		${CURAND_DLL}
		DESTINATION bin)

	file(TO_CMAKE_PATH "${CUDNN_HOME}/bin/cudnn64_8.dll" CUDNN_DLL)
	file(TO_CMAKE_PATH "${CUDNN_HOME}/bin/cudnn_adv_infer64_8.dll" CUDNN_ADV_INFER_DLL)
	file(TO_CMAKE_PATH "${CUDNN_HOME}/bin/cudnn_cnn_infer64_8.dll" CUDNN_CNN_INFER_DLL)
	file(TO_CMAKE_PATH "${CUDNN_HOME}/bin/cudnn_ops_infer64_8.dll" CUDNN_OPS_INFER_DLL)
	install(PROGRAMS
		${CUDNN_DLL}
		${CUDNN_ADV_INFER_DLL}
		${CUDNN_CNN_INFER_DLL}
		${CUDNN_OPS_INFER_DLL}
		DESTINATION bin)
endif()

list(APPEND CPACK_WIX_CANDLE_EXTRA_FLAGS "-dCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
list(APPEND CPACK_WIX_CANDLE_EXTRA_FLAGS "-dCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

set(CPACK_WIX_EXTENSIONS "WixDifxAppExtension")
set(CPACK_WIX_CUSTOM_XMLNS "difx=http://schemas.microsoft.com/wix/DifxAppExtension")

set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patch.xml")

add_package(testo-nn-server-package)
add_dependencies(testo-nn-server-package testo_nn_server)

if (USE_CUDA)
	install(TARGETS testo_request_license)
	add_dependencies(testo-nn-server-package testo_request_license)
endif()
