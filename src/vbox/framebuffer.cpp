
#include "framebuffer.hpp"
#include <cassert>
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
	if (iid == IID_IMarshal) {
		return IMarshal_QueryInterface(((IFramebuffer*)self)->marshal, iid, result);
	}
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
	if (!memcmp(iid, &IID_IFramebuffer, sizeof(nsIID))) {
		AddRef(self);
		*result = self;
		return S_OK;
	}
	*result = nullptr;
	return E_NOINTERFACE;
}
#endif

static HRESULT get_Width(::IFramebuffer* self, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_Height(::IFramebuffer* self, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_BitsPerPixel(::IFramebuffer* self, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_BytesPerLine(::IFramebuffer* self, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_PixelFormat(::IFramebuffer* self, BitmapFormat_T* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_HeightReduction(::IFramebuffer* self, ULONG* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_Overlay(::IFramebuffer* self, IFramebufferOverlay** result) {
	assert(false);
	return S_OK;
}
static HRESULT get_WinId(::IFramebuffer* self, LONG64* result) {
	assert(false);
	return S_OK;
}
static HRESULT get_Capabilities(::IFramebuffer* self, SAFEARRAY_OUT_PARAM(FramebufferCapabilities_T, result)) {
	try {
		FramebufferCapabilities capabilities = FramebufferCapabilities_UpdateImage;
		SafeArray safe_array = SafeArray::bitset(capabilities);
		SAFEARRAY_MOVE_TO_OUT_PARAM(safe_array, result);
		return S_OK;
	} catch (const std::exception&) {
		return E_UNEXPECTED;
	}
}
static HRESULT NotifyUpdate(::IFramebuffer* self, ULONG x, ULONG y, ULONG width, ULONG height) {
	printf("NotifyUpdate\n");
	return S_OK;
}
static HRESULT NotifyUpdateImage(::IFramebuffer* self, ULONG x, ULONG y, ULONG width, ULONG height, SAFEARRAY_IN_PARAM(uint8_t, image)) {
	printf("NotifyUpdateImage\n");
	return S_OK;
}
static HRESULT NotifyChange(::IFramebuffer* self, ULONG screen_id, ULONG x_origin, ULONG y_origin, ULONG width, ULONG height) {
	printf("NotifyChange\n");
	return S_OK;
}
static HRESULT VideoModeSupported(::IFramebuffer* self, ULONG width, ULONG height, ULONG, BOOL* supported) {
	assert(false);
	return S_OK;
}
static HRESULT GetVisibleRegion(::IFramebuffer* self, uint8_t* rectangles, ULONG count, ULONG* count_copied) {
	assert(false);
	return S_OK;
}
static HRESULT SetVisibleRegion(::IFramebuffer* self, uint8_t* rectangles, ULONG count) {
	assert(false);
	return S_OK;
}
static HRESULT ProcessVHWACommand(::IFramebuffer* self, uint8_t* command) {
	assert(false);
	return S_OK;
}
static HRESULT Notify3DEvent(::IFramebuffer* self, ULONG type, SAFEARRAY_IN_PARAM(uint8_t, data)) {
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
