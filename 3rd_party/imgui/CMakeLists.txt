cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(SRCS
	imgui.cpp
	imgui_draw.cpp
	imgui_widgets.cpp
)
if(WIN32)
	set(IMPL_SRCS
		imgui_impl_dx11.cpp
		imgui_impl_win32.cpp
	)
elseif(APPLE)
	set(IMPL_SRCS
		imgui_impl_metal.mm
		imgui_impl_osx.mm
	)
	set_property(SOURCE ${IMPL_SRCS} PROPERTY COMPILE_FLAGS -fobjc-weak)
elseif(UNIX)
	include_directories(${CMAKE_SOURCE_DIR}/3rd_party/gl3w)
	set(IMPL_SRCS
		imgui_impl_opengl3.cpp
		imgui_impl_glfw.cpp
	)
endif()
add_library(imgui STATIC ${SRCS} ${IMPL_SRCS})
