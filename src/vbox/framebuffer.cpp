
#include "framebuffer.hpp"
#include <cassert>
#include <set>
#include <cstring>
#include "throw_if_failed.hpp"
#include "safe_array.hpp"

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

#ifdef WIN32
static HRESULT QueryInterface(::IFramebuffer* self, const IID& iid, void** result) {
	if (iid == IID_IFramebuffer || iid == IID_IUnknown) {
		AddRef(self);
		*result = self;
		return S_OK;
	}
	*result = nullptr;
	return E_NOINTERFACE;
}
#else
static HRESULT QueryInterface(::IFramebuffer* self, const nsIID* iid, void** result) {
	const nsIID IID_nsISupports = NS_ISUPPORTS_IID;
	if (!memcmp(iid, &IID_IFramebuffer, sizeof(nsIID)) || !memcmp(iid, &IID_nsISupports, sizeof(nsIID))) {
		AddRef(self);
		*result = self;
		return S_OK;
	}
	*result = nullptr;
	return E_NOINTERFACE;
}
#endif

static HRESULT get_Width(::IFramebuffer* self, ULONG* width) {
	assert(false);
	return 0;
}
static HRESULT get_Height(::IFramebuffer* self, ULONG* height) {
	assert(false);
	return 0;
}
static HRESULT get_BitsPerPixel(::IFramebuffer* self, ULONG* bits_per_pixel) {
	assert(false);
	return 0;
}
static HRESULT get_BytesPerLine(::IFramebuffer* self, ULONG* bytes_per_line) {
	assert(false);
	return 0;
}
static HRESULT get_PixelFormat(::IFramebuffer* self, BitmapFormat_T* pixel_format) {
	assert(false);
	return 0;
}
static HRESULT get_HeightReduction(::IFramebuffer* self, ULONG* height_reduction) {
	assert(false);
	return 0;
}
static HRESULT get_Overlay(::IFramebuffer* self, IFramebufferOverlay** overlay) {
	assert(false);
	return 0;
}
static HRESULT get_WinId(::IFramebuffer* self, LONG64* win_id) {
	assert(false);
	return 0;
}
static HRESULT get_Capabilities(::IFramebuffer* self, SAFEARRAY_OUT_PARAM(FramebufferCapabilities_T, out)) {
	try {
		FramebufferCapabilities capabilities = FramebufferCapabilities_UpdateImage;
		SafeArray safe_array = SafeArray::bitset(capabilities);
		SAFEARRAY_MOVE_TO_OUT_PARAM(safe_array, out);
		return S_OK;
	} catch (const std::exception&) {
		return E_UNEXPECTED;
	}
}
static HRESULT NotifyUpdate(::IFramebuffer* self, ULONG x, ULONG y, ULONG width, ULONG height) {
	assert(false);
	return 0;
}
static HRESULT NotifyUpdateImage(::IFramebuffer* self, ULONG x, ULONG y, ULONG width, ULONG height, SAFEARRAY_IN_PARAM(uint8_t, in)) {
	SAFEARRAY handle;
	ISafeArray safe_array = {&handle};
	SAFEARRAY_MOVE_FROM_IN_PARAM(safe_array, in);
	assert(false);
	return 0;
}
static HRESULT NotifyChange(::IFramebuffer* self, ULONG screen_id, ULONG x_origin, ULONG y_origin, ULONG width, ULONG height) {
	assert(false);
	return 0;
}
static HRESULT VideoModeSupported(::IFramebuffer* self, ULONG width, ULONG height, ULONG, BOOL* supported) {
	assert(false);
	return 0;
}
static HRESULT GetVisibleRegion(::IFramebuffer* self, uint8_t* rectangles, ULONG count, ULONG* count_copied) {
	assert(false);
	return 0;
}
static HRESULT SetVisibleRegion(::IFramebuffer* self, uint8_t* rectangles, ULONG count) {
	assert(false);
	return 0;
}
static HRESULT ProcessVHWACommand(::IFramebuffer* self, uint8_t* command) {
	assert(false);
	return 0;
}
static HRESULT Notify3DEvent(::IFramebuffer* self, ULONG type, SAFEARRAY_IN_PARAM(uint8_t, data)) {
	assert(false);
	return 0;
}

static IFramebufferVtbl framebuffer_vtbl = {
	QueryInterface,
	AddRef,
	Release,
#ifdef WIN32
	nullptr,
	nullptr,
	nullptr,
	nullptr,
#endif
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
