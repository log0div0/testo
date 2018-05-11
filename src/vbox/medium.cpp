
#include "medium.hpp"
#include <stdexcept>
#include <ostream>
#include "throw_if_failed.hpp"
#include "string.hpp"
#include "safe_array.hpp"

namespace vbox {

Medium::Medium(IMedium* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Medium::~Medium() {
	if (handle) {
		IMedium_Release(handle);
	}
}

Medium::Medium(Medium&& other): handle(other.handle) {
	other.handle = nullptr;
}

Medium& Medium::operator=(Medium&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Medium::name() const {
	try {
		BSTR name = nullptr;
		throw_if_failed(IMedium_get_Name(handle, &name));
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

MediumState Medium::state() const {
	try {
		MediumState_T state = MediumState_NotCreated;
		throw_if_failed(IMedium_get_State(handle, &state));
		return (MediumState)state;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

MediumState Medium::refresh_state() const {
	try {
		MediumState_T state = MediumState_NotCreated;
		throw_if_failed(IMedium_RefreshState(handle, &state));
		return (MediumState)state;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

MediumVariant Medium::variant() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IMedium_get_Variant(handle, ComSafeArrayAsOutTypeParam(safe_array.handle, MediumVariant_T)));
		return (MediumVariant)safe_array.bitset();
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Medium::create_base_storage(size_t size, MediumVariant variant) {
	try {
		SafeArray safe_array = SafeArray::bitset(variant);

		IProgress* result = nullptr;
		throw_if_failed(IMedium_CreateBaseStorage(handle,
			size,
			ComSafeArrayAsInParam(safe_array.handle, MediumVariant_T),
			&result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Medium::operator bool() const {
	return handle != nullptr;
}

std::ostream& operator<<(std::ostream& stream, const Medium& medium) {
	stream << "name=" << medium.name();
	stream << " variant=" << medium.variant();
	stream << " state=" << medium.state();
	return stream;
}

}
