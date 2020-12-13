
#include "WbemObjectTextSrc.hpp"
#include "Error.hpp"
#include <comutil.h>

namespace wmi {

WbemObjectTextSrc::WbemObjectTextSrc() {
	try {
		throw_if_failed(CoCreateInstance(
			CLSID_WbemObjectTextSrc,
			nullptr,
			CLSCTX_INPROC_SERVER,
			IID_IWbemObjectTextSrc,
			(void**)&handle));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string WbemObjectTextSrc::getText(const WbemClassObject& object, ULONG format) {
	try {
		BSTR str = nullptr;
		throw_if_failed(handle->GetText(0,
			object.handle,
			WMI_OBJ_TEXT_CIM_DTD_2_0,
			nullptr,
			&str));
		return (const char*)bstr_t(str, false);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
