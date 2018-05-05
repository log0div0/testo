cmake_minimum_required(VERSION 2.8)

if (WIN32)
	set(FILES VirtualBox_i_mscom.c)
else()
	set(FILES VirtualBox_i_xpcom.c)
endif()

add_library(vboxcapi STATIC ${FILES})
