
#include "framebuffer.hpp"
#include <cassert>
#include "throw_if_failed.hpp"

namespace vbox {

static ULONG AddRef(::IFramebuffer* self) {
	auto fb = (IFramebuffer*)self;
	fb->refcnt++;
	return fb->refcnt;
}
static ULONG Release(::IFramebuffer* self) {
	auto fb = (IFramebuffer*)self;
	fb->refcnt--;
	if (fb->refcnt == 0) {
		delete fb;
		return 0;
	};
	return fb->refcnt;
}

static HRESULT QueryInterface(::IFramebuffer* self, const IID& iid, void** result) {
	if (iid == IID_IFramebuffer || iid == IID_IUnknown) {
		AddRef(self);
		*result = self;
		return S_OK;
	}
	*result = nullptr;
	return E_NOINTERFACE;
}
static HRESULT get_Width(::IFramebuffer* self, ULONG *width) {
	assert(false);
	return 0;
}
static HRESULT get_Height(::IFramebuffer* self, ULONG *height) {
	assert(false);
	return 0;
}
static HRESULT get_BitsPerPixel(::IFramebuffer* self, ULONG *bitsPerPixel) {
	assert(false);
	return 0;
}
static HRESULT get_BytesPerLine(::IFramebuffer* self, ULONG *bytesPerLine) {
	assert(false);
	return 0;
}
static HRESULT get_PixelFormat(::IFramebuffer* self, BitmapFormat *pixelFormat) {
	assert(false);
	return 0;
}
static HRESULT get_HeightReduction(::IFramebuffer* self, ULONG *heightReduction) {
	assert(false);
	return 0;
}
static HRESULT get_Overlay(::IFramebuffer* self, IFramebufferOverlay * *overlay) {
	assert(false);
	return 0;
}
static HRESULT get_WinId(::IFramebuffer* self, LONG64 *winId) {
	assert(false);
	return 0;
}
static HRESULT get_Capabilities(::IFramebuffer* self, SAFEARRAY **capabilities) {
	assert(false);
	return 0;
}
static HRESULT NotifyUpdate(::IFramebuffer* self, ULONG x, ULONG y, ULONG width, ULONG height) {
	assert(false);
	return 0;
}
static HRESULT NotifyUpdateImage(::IFramebuffer* self, ULONG x, ULONG y, ULONG width, ULONG height, SAFEARRAY* image) {
	assert(false);
	return 0;
}
static HRESULT NotifyChange(::IFramebuffer* self, ULONG screenId, ULONG xOrigin, ULONG yOrigin, ULONG width, ULONG height) {
	assert(false);
	return 0;
}
static HRESULT VideoModeSupported(::IFramebuffer* self, ULONG width, ULONG height, ULONG bpp, BOOL * supported) {
	assert(false);
	return 0;
}
static HRESULT GetVisibleRegion(::IFramebuffer* self, uint8_t * rectangles, ULONG count, ULONG * countCopied) {
	assert(false);
	return 0;
}
static HRESULT SetVisibleRegion(::IFramebuffer* self, uint8_t * rectangles, ULONG count) {
	assert(false);
	return 0;
}
static HRESULT ProcessVHWACommand(::IFramebuffer* self, uint8_t * command) {
	assert(false);
	return 0;
}
static HRESULT Notify3DEvent(::IFramebuffer* self, ULONG type, SAFEARRAY* data) {
	assert(false);
	return 0;
}

static IFramebufferVtbl framebuffer_vtbl = {
	QueryInterface,
	AddRef,
	Release,
	nullptr,
	nullptr,
	nullptr,
	nullptr,
	get_Width,
	get_Height,
	get_BitsPerPixel,
	get_BytesPerLine,
	get_PixelFormat,
	get_HeightReduction,
	get_Overlay,
	get_WinId,
	get_Capabilities,
	NotifyUpdate,
	NotifyUpdateImage,
	NotifyChange,
	VideoModeSupported,
	GetVisibleRegion,
	SetVisibleRegion,
	ProcessVHWACommand,
	Notify3DEvent,
};

IFramebuffer::IFramebuffer(): ::IFramebuffer {&framebuffer_vtbl} {

}

Framebuffer::Framebuffer(IFramebuffer* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Framebuffer::~Framebuffer() {
	if (handle) {
		IFramebuffer_Release(handle);
	}
}

Framebuffer::Framebuffer(Framebuffer&& other): handle(other.handle) {
	other.handle = nullptr;
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other) {
	std::swap(handle, other.handle);
	return *this;
}

}
