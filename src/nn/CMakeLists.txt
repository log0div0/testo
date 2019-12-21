cmake_minimum_required(VERSION 3.0)

include_directories(
	${ONNX_RUNTIME_DIR}/include
)

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
add_subdirectory(text_detector)
