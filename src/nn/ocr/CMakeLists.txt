cmake_minimum_required(VERSION 3.0)

file(GLOB SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

embed_onnx_model(${CMAKE_CURRENT_SOURCE_DIR}/TextDetector.onnx)
embed_onnx_model(${CMAKE_CURRENT_SOURCE_DIR}/TextRecognizer.onnx)

add_library(ocr STATIC ${SRCS})
target_link_libraries(ocr stb onnxruntime pthread)

add_executable(ocr_demo main.cpp)
target_link_libraries(ocr_demo ocr)

file(GLOB_RECURSE TEST_SRCS tests/*.cpp)
add_executable(ocr_tests ${TEST_SRCS})
target_link_libraries(ocr_tests ocr)

add_custom_target(run-ocr-tests
	COMMAND ${CMAKE_BINARY_DIR}/out/sbin/ocr_tests --durations yes
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
	DEPENDS ocr_tests
)
