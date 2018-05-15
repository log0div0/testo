
#pragma once

#include "api.hpp"

namespace vbox {

struct IFramebuffer: ::IFramebuffer {
	IFramebuffer();
	virtual ~IFramebuffer();

	virtual void notify_update(ULONG x, ULONG y, ULONG width, ULONG height) {};
	virtual void notify_change(ULONG screen_id, ULONG x_origin, ULONG y_origin, ULONG width, ULONG height) {};

	ULONG refcnt = 1;
#ifdef WIN32
	IMarshal* marshal = nullptr;
#endif
	FramebufferCapabilities capabilities = (FramebufferCapabilities)0;
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
