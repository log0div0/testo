cmake_minimum_required(VERSION 3.0)

set(CPACK_PACKAGE_NAME "Testo")
set(CPACK_PACKAGE_VERSION "${TESTO_VERSION}")
set(CPACK_WIX_UPGRADE_GUID "D80951FA-43DF-4B00-87CC-43A2824620D6")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Testo Lang/Testo")

file(TO_CMAKE_PATH "${ONNX_RUNTIME_DIR}/bin/onnxruntime.dll" ONNX_RUNTIME_DLL)

install(TARGETS testo)
install(PROGRAMS ${ONNX_RUNTIME_DLL} DESTINATION bin)

install(FILES
	${CMAKE_SOURCE_DIR}/src/nn/Homm3Detector.onnx
	${CMAKE_SOURCE_DIR}/src/nn/TextColorPicker.onnx
	${CMAKE_SOURCE_DIR}/src/nn/TextDetector.onnx
	${CMAKE_SOURCE_DIR}/src/nn/TextRecognizer.onnx
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
	install(PROGRAMS
		${CUDNN_DLL}
		DESTINATION bin)
endif()

set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patch.xml")

add_package(testo-package)
add_dependencies(testo-package testo)
