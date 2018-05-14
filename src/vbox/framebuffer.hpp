
#pragma once

#include "api.hpp"

namespace vbox {

struct IFramebuffer: ::IFramebuffer {
	IFramebuffer();
	~IFramebuffer();
	ULONG refcnt = 1;
#ifdef WIN32
	IMarshal* marshal = nullptr;
#endif
};

struct Framebuffer {
	Framebuffer(IFramebuffer* handle);
	~Framebuffer();

	Framebuffer(const Framebuffer&) = delete;
	Framebuffer& operator=(const Framebuffer&) = delete;
	Framebuffer(Framebuffer&& other);
	Framebuffer& operator=(Framebuffer&& other);

	IFramebuffer* handle = nullptr;
};

}
