
#include "framebuffer.hpp"
#include <cassert>
#include <cstring>
#include "throw_if_failed.hpp"
#include "safe_array.hpp"

namespace vbox {

static ULONG AddRef(::IFramebuffer* framebuffer) {
	auto fb = (IFramebuffer*)framebuffer;
	fb->refcnt++;
	return fb->refcnt;
}
static ULONG Release(::IFramebuffer* framebuffer) {
	auto fb = (IFramebuffer*)framebuffer;
	fb->refcnt--;
	if (fb->refcnt == 0) {
		delete fb;
		return 0;
	};
	return fb->refcnt;
}

#ifdef WIN32
static HRESULT QueryInterface(::IFramebuffer* framebuffer, const IID& iid, void** result) {
	if (iid == IID_IMarshal) {
		return IMarshal_QueryInterface(((IFramebuffer*)framebuffer)->marshal, iid, result);
	}
	if (iid == IID_IFramebuffer || iid == IID_IUnknown) {
		AddRef(framebuffer);
		*result = framebuffer;
		return S_OK;
	}
	*result = nullptr;
	return E_NOINTERFACE;
}
#else
static HRESULT QueryInterface(::IFramebuffer* framebuffer, const nsIID* iid, void** result) {
	if (!memcmp(iid, &IID_IFramebuffer, sizeof(nsIID))) {
		AddRef(framebuffer);
		*result = framebuffer;
		return S_OK;
	}
	*result = nullptr;
	return E_NOINTERFACE;
}
#endif

static HRESULT get_Width(::IFramebuffer* framebuffer, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_Height(::IFramebuffer* framebuffer, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_BitsPerPixel(::IFramebuffer* framebuffer, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_BytesPerLine(::IFramebuffer* framebuffer, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_PixelFormat(::IFramebuffer* framebuffer, BitmapFormat_T* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_HeightReduction(::IFramebuffer* framebuffer, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_Overlay(::IFramebuffer* framebuffer, IFramebufferOverlay** result) {
	assert(false);
	return S_OK;
}
static HRESULT get_WinId(::IFramebuffer* framebuffer, LONG64* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_Capabilities(::IFramebuffer* framebuffer, SAFEARRAY_OUT_PARAM(FramebufferCapabilities_T, result)) {
	try {
		SafeArray safe_array;
		safe_array = SafeArray::bitset(((IFramebuffer*)framebuffer)->capabilities);
		SAFEARRAY_MOVE_TO_OUT_PARAM(safe_array, result);
		return S_OK;
	} catch (const std::exception&) {
		return E_UNEXPECTED;
	}
}
static HRESULT NotifyUpdate(::IFramebuffer* framebuffer, ULONG x, ULONG y, ULONG width, ULONG height) {
	try {
		((IFramebuffer*)framebuffer)->notify_update(x, y, width, height);
		return S_OK;
	} catch (const std::exception&) {
		return E_UNEXPECTED;
	}
}
static HRESULT NotifyUpdateImage(::IFramebuffer* framebuffer, ULONG x, ULONG y, ULONG width, ULONG height, SAFEARRAY_IN_PARAM(uint8_t, image)) {
	assert(false);
	return S_OK;
}
static HRESULT NotifyChange(::IFramebuffer* framebuffer, ULONG screen_id, ULONG x_origin, ULONG y_origin, ULONG width, ULONG height) {
	try {
		((IFramebuffer*)framebuffer)->notify_change(screen_id, x_origin, y_origin, width, height);
		return S_OK;
	} catch (const std::exception&) {
		return E_UNEXPECTED;
	}
}
static HRESULT VideoModeSupported(::IFramebuffer* framebuffer, ULONG width, ULONG height, ULONG, BOOL* supported) {
	assert(false);
	return S_OK;
}
static HRESULT GetVisibleRegion(::IFramebuffer* framebuffer, uint8_t* rectangles, ULONG count, ULONG* count_copied) {
	assert(false);
	return S_OK;
}
static HRESULT SetVisibleRegion(::IFramebuffer* framebuffer, uint8_t* rectangles, ULONG count) {
	assert(false);
	return S_OK;
}
static HRESULT ProcessVHWACommand(::IFramebuffer* framebuffer, uint8_t* command) {
	assert(false);
	return S_OK;
}
static HRESULT Notify3DEvent(::IFramebuffer* framebuffer, ULONG type, SAFEARRAY_IN_PARAM(uint8_t, data)) {
	assert(false);
	return S_OK;
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
	try {
#ifdef WIN32
		HRESULT rc = CoCreateFreeThreadedMarshaler((IUnknown*)this, (IUnknown**)&marshal);
		if (FAILED(rc)) {
			throw std::runtime_error("CoCreateFreeThreadedMarshaler");
		}
#endif
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

IFramebuffer::~IFramebuffer() {
#ifdef WIN32
	IMarshal_Release(marshal);
#endif
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
