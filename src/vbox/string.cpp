
#include "string.hpp"
#include "throw_if_failed.hpp"

namespace vbox {

StringIn::StringIn(const std::string& str) {
	try {
		throw_if_failed(api->pfnUtf8ToUtf16(str.c_str(), &data));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

StringIn::~StringIn() {
	if (data) {
		api->pfnUtf16Free(data);
	}
}

StringIn::StringIn(StringIn&& other): data(other.data) {
	other.data = nullptr;
}

StringIn::operator BSTR() const {
	return data;
}

StringOut::StringOut(BSTR str) {
	try {
		if (!str) {
			throw std::runtime_error("Invalid argument");
		}
		throw_if_failed(api->pfnUtf16ToUtf8(str, &data));
		api->pfnComUnallocString(str);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

StringOut::~StringOut() {
	if (data) {
		api->pfnUtf8Free(data);
	}
}

StringOut::StringOut(StringOut&& other): data(other.data) {
	other.data = nullptr;
}

StringOut::operator std::string() const {
	return data;
}

}
