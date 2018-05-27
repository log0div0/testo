
#pragma once

#include "safe_array.hpp"
#include <mutex>

namespace vbox {

struct IFramebuffer: ::IFramebuffer {
	IFramebuffer();
	virtual ~IFramebuffer();

	virtual FramebufferCapabilities capabilities() const = 0;
	virtual void notify_change(ULONG screen_id, ULONG x_origin, ULONG y_origin, ULONG width, ULONG height) = 0;
	virtual void notify_update_image(ULONG x, ULONG y, ULONG width, ULONG height, const void* image) = 0;

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
