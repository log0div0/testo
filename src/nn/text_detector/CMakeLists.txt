cmake_minimum_required(VERSION 3.0)

set(SRCS TextDetector.cpp)
embed_ftlite_model(${CMAKE_CURRENT_SOURCE_DIR}/TextDetector.tflite)

add_library(text-detector STATIC ${SRCS})
target_link_libraries(text-detector stb tensorflow-lite)

add_executable(text_detector_demo demo.cpp)
target_link_libraries(text_detector_demo text-detector)

add_executable(text_detector_tests tests.cpp)
target_link_libraries(text_detector_tests text-detector)

add_custom_target(run_text_detector_tests
	COMMAND ${CMAKE_BINARY_DIR}/out/sbin/text_detector_tests
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	DEPENDS text_detector_tests
)
