
#include "WbemClassObject.hpp"
#include "Error.hpp"
#include <comutil.h>

namespace wmi {

Variant WbemClassObject::get(const std::string& name) const {
	try {
		Variant variant;
		throw_if_failed(handle->Get(bstr_t(name.c_str()), 0, &variant, nullptr, 0));
		return variant;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string WbemClassObject::getObjectText() {
	try {
		BSTR str = nullptr;
		throw_if_failed(handle->GetObjectText(0, &str));
		return (const char*)bstr_t(str, false);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string WbemClassObject::relpath() const {
	try {
		return get("__RELPATH");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string WbemClassObject::path() const {
	try {
		return get("__PATH");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

WbemClassObject WbemClassObject::getMethod(const std::string& name) const {
	try {
		IWbemClassObject* object = nullptr;
		throw_if_failed(handle->GetMethod(
			bstr_t(name.c_str()),
			0,
			&object,
			nullptr
		));
		if (object) {
			return WbemClassObject(object);
		} else {
			return WbemClassObject();
		}
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

WbemClassObject WbemClassObject::spawnInstance() const {
	try {
		IWbemClassObject* object = nullptr;
		throw_if_failed(handle->SpawnInstance(
			0,
			&object
		));
		return object;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

WbemClassObject& WbemClassObject::put(const std::string& name, const Variant& value, CIMTYPE type) {
	try {
		throw_if_failed(handle->Put(
			bstr_t(name.c_str()),
			0,
			(VARIANT*)&value,
			type
		));
		return *this;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

WbemClassObject WbemClassObject::clone() {
	try {
		IWbemClassObject* copy = nullptr;
		throw_if_failed(handle->Clone(&copy));
		return copy;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
