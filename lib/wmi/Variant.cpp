
#include "Variant.hpp"
#include "Error.hpp"
#include <propvarutil.h>
#include <comutil.h>

namespace wmi {

Variant::Variant() {
	VariantInit(this);
}

Variant::Variant(bool value) {
	InitVariantFromBoolean(value, this);
}

Variant::Variant(const char* str) {
	InitVariantFromString(bstr_t(str), this);
}

Variant::Variant(const std::string& str) {
	InitVariantFromString(bstr_t(str.c_str()), this);
}

Variant::Variant(const std::vector<uint8_t>& buffer) {
	InitVariantFromBuffer(buffer.data(), buffer.size(), this);
}

Variant::Variant(const std::vector<std::string>& strs) {
	std::vector<bstr_t> bstrs;
	bstrs.reserve(strs.size());
	for (auto& str: strs) {
		bstrs.push_back(str.c_str());
	}
	std::vector<PCWSTR> pcwstrs;
	pcwstrs.reserve(pcwstrs.size());
	for (auto& bstr: bstrs) {
		pcwstrs.push_back(bstr);
	}
	InitVariantFromStringArray(pcwstrs.data(), pcwstrs.size(), this);
}

Variant::Variant(uint16_t value) {
	InitVariantFromUInt16(value, this);
}

Variant::Variant(int16_t value) {
	InitVariantFromInt16(value, this);
}

Variant::Variant(uint32_t value) {
	InitVariantFromUInt32(value, this);
}

Variant::Variant(int32_t value) {
	InitVariantFromInt32(value, this);
}

Variant::Variant(uint64_t value) {
	InitVariantFromUInt64(value, this);
}

Variant::Variant(int64_t value) {
	InitVariantFromInt64(value, this);
}

Variant::~Variant() {
	VariantClear(this);
}

Variant::Variant(Variant&& other): VARIANT(other) {
	VariantInit(&other);
}

Variant& Variant::operator=(Variant&& other) {
	std::swap((VARIANT&)*this, (VARIANT&)other);
	return *this;
}

Variant::operator std::string() const {
	try {
		check_type(VT_BSTR);
		if (bstrVal == nullptr) {
			throw std::runtime_error("nullptr");
		}
		return (const char*)bstr_t(bstrVal);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Variant::operator std::vector<uint8_t>() const {
	try {
		check_type(VARENUM(VT_ARRAY | VT_UI1));
		std::vector<uint8_t> result(VariantGetElementCount(*this));
		throw_if_failed(VariantToBuffer(*this, result.data(), result.size()));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Variant::operator std::vector<std::string>() const
{
	try {
		check_type(VARENUM(VT_BSTR | VT_ARRAY));
		std::vector<PWSTR> pwstrs(VariantGetElementCount(*this));
		ULONG size = 0;
		throw_if_failed(VariantToStringArray(*this, pwstrs.data(), pwstrs.size(), &size));
		if (pwstrs.size() != size) {
			throw std::runtime_error("Extracted less strings than expected");
		}
		std::vector<bstr_t> bstrs;
		bstrs.reserve(pwstrs.size());
		for (auto& pwstr: pwstrs) {
			bstrs.push_back(bstr_t(pwstr));
			CoTaskMemFree(pwstr);
		}
		std::vector<std::string> strs;
		strs.reserve(bstrs.size());
		for (auto& bstr: bstrs) {
			strs.push_back((const char*)bstr);
		}
		return strs;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Variant::operator uint16_t() const {
	try {
		check_type(VT_UI2);
		return uiVal;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Variant::operator int16_t() const {
	try {
		check_type(VT_I2);
		return iVal;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Variant::operator uint32_t() const {
	try {
		check_type(VT_UI4);
		return ulVal;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Variant::operator int32_t() const {
	try {
		check_type(VT_I4);
		return lVal;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Variant::check_type(VARENUM expected) const {
	if (vt != expected) {
		throw std::runtime_error("Expected type = " + to_hex(expected) + " , actual type = " + to_hex(vt));
	}
}

}
