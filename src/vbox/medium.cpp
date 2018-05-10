
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

std::set<MediumVariant> Medium::variant() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IMedium_get_Variant(handle, SAFEARRAY_AS_OUT_PARAM(MediumVariant_T, safe_array)));
		ArrayOut array_out = safe_array.copy_out(VT_I4);
		std::set<MediumVariant> result;
		for (ULONG i = 0; i < array_out.values_count / sizeof(int); ++i) {
			result.insert(((MediumVariant*)array_out.values)[i]);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress Medium::create_base_storage(size_t size, std::set<MediumVariant> variants) {
	try {
		std::vector<MediumVariant> vector(sizeof(MediumVariant) * 8, MediumVariant_Standard);
		for (auto& variant: variants) {
			vector[variant] = variant;
		}

		SafeArray safe_array(VT_I4, (ULONG)vector.size());
		safe_array.copy_in(vector.data(), (ULONG)(vector.size() * sizeof(MediumVariant)));

		IProgress* result = nullptr;
		throw_if_failed(IMedium_CreateBaseStorage(handle,
			size,
			SAFEARRAY_AS_IN_PARAM(MediumVariant_T, safe_array),
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
	stream << " variant=";
	size_t i = 0;
	for (auto& variant: medium.variant()) {
		if (i != 0) {
			stream << "|";
		}
		stream << variant;
		++i;
	}
	stream << " state=" << medium.state();
	return stream;
}

}
