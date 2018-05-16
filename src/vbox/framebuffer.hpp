
#pragma once

#include "safe_array.hpp"

namespace vbox {

struct IFramebuffer: ::IFramebuffer {
	IFramebuffer();
	~IFramebuffer();

	void notify_change(ULONG screen_id, ULONG x_origin, ULONG y_origin, ULONG width, ULONG height);
	void notify_update(ULONG x, ULONG y, ULONG width, ULONG height);
	void notify_update_image(ULONG x, ULONG y, ULONG width, ULONG height, const SafeArrayView& image);

	ULONG refcnt = 1;
#ifdef WIN32
	IMarshal* marshal = nullptr;
#endif
	FramebufferCapabilities capabilities = FramebufferCapabilities_UpdateImage;
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
