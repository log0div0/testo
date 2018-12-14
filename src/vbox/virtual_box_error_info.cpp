
#include <vbox/virtual_box_error_info.hpp>
#include <vbox/string.hpp>

namespace vbox {

VirtualBoxErrorInfo::VirtualBoxErrorInfo(IVirtualBoxErrorInfo* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

VirtualBoxErrorInfo::~VirtualBoxErrorInfo() {
	if (handle) {
		IVirtualBoxErrorInfo_Release(handle);
	}
}

std::string VirtualBoxErrorInfo::text() const {
	BSTR result = nullptr;
	HRESULT rc = IVirtualBoxErrorInfo_get_Text(handle, &result);
	if (FAILED(rc)) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
	return StringOut(result);
}

VirtualBoxErrorInfo::operator bool() const {
	return handle != nullptr;
}

}
