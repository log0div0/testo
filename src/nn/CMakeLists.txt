cmake_minimum_required(VERSION 3.0)

macro(embed_onnx_model MODEL_PATH)
	get_filename_component(MODEL_NAME ${MODEL_PATH} NAME)
	get_filename_component(MODEL_DIRECTORY ${MODEL_PATH} DIRECTORY)
	add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/${MODEL_NAME}.cpp
		WORKING_DIRECTORY ${MODEL_DIRECTORY}
		COMMAND xxd -i ${MODEL_NAME} ${CMAKE_BINARY_DIR}/src/${MODEL_NAME}.cpp
		DEPENDS ${MODEL_PATH}
	)
	list(APPEND SRCS ${CMAKE_BINARY_DIR}/src/${MODEL_NAME}.cpp)
endmacro(embed_onnx_model)

file(GLOB SRCS *.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

embed_onnx_model(${CMAKE_CURRENT_SOURCE_DIR}/TextDetector.onnx)
embed_onnx_model(${CMAKE_CURRENT_SOURCE_DIR}/TextRecognizer.onnx)
embed_onnx_model(${CMAKE_CURRENT_SOURCE_DIR}/TextColorPicker.onnx)

add_library(nn STATIC ${SRCS})
target_link_libraries(nn stb)

set(CMAKE_EXE_LINKER_FLAGS "${ONNX_RUNTIME_LINKER_FLAGS}")

if(UNIX)
	target_link_libraries(nn pthread dl)
endif()

add_executable(nn_demo main.cpp)
target_link_libraries(nn_demo nn)

file(GLOB_RECURSE TEST_SRCS tests/*.cpp)
add_executable(nn_tests ${TEST_SRCS})
target_link_libraries(nn_tests nn)

if(UNIX)
add_custom_target(run-nn-tests
	COMMAND ${CMAKE_BINARY_DIR}/out/sbin/nn_tests --durations yes
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
	DEPENDS nn_tests
)
endif()
