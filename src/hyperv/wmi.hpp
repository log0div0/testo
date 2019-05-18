
#pragma once

#include <vector>
#include <string>
#include <locale>
#include <codecvt>

#include <Wbemidl.h>
#include <comdef.h>

namespace wmi {

extern std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;

struct CoInitializer {
	CoInitializer() {
		HRESULT hres = CoInitializeEx(0, COINIT_MULTITHREADED);
		if (FAILED(hres)) {
			throw std::runtime_error("CoInitializeEx failed");
		}
	}

	void initalize_security() {
		HRESULT hres = CoInitializeSecurity(
			NULL,
			-1,
			NULL,
			NULL,
			RPC_C_AUTHN_LEVEL_DEFAULT,
			RPC_C_IMP_LEVEL_IMPERSONATE,
			NULL,
			EOAC_NONE,
			NULL
		);
		if (FAILED(hres)) {
			throw std::runtime_error("CoInitializeSecurity failed");
		}
	}

	~CoInitializer() {
		CoUninitialize();
	}
};

struct Variant: VARIANT {
	Variant() {
		VariantInit(this);
	}
	~Variant() {
		VariantClear(this);
	}

	Variant(const Variant& other) = delete;
	Variant& operator=(const Variant& other) = delete;
	Variant(Variant&& other): VARIANT(other) {
		VariantInit(&other);
	}
	Variant& operator=(Variant&& other) {
		std::swap((VARIANT&)*this, (VARIANT&)other);
		return *this;
	}
};

struct String {
	String(BSTR handle_): handle(handle_) {
		if (handle == nullptr) {
			throw std::runtime_error("null bstr");
		}
	}
	~String() {
		if (handle) {
			SysFreeString(handle);
			handle = nullptr;
		}
	}
	operator std::string() const {
		return converter.to_bytes(handle);
	}
	BSTR handle = nullptr;
};

template <typename IObject>
struct Object {
	Object() {}
	Object(IObject* handle_): handle(handle_) {
		if (handle == nullptr) {
			throw std::runtime_error("IObject handle == nullptr");
		}
	}
	~Object() {
		if (handle) {
			handle->Release();
			handle = nullptr;
		}
	}

	Object(const Object& other) = delete;
	Object& operator=(const Object& other) = delete;
	Object(Object&& other): handle(other.handle) {
		other.handle = nullptr;
	}
	Object& operator=(Object&& other) {
		std::swap(handle, other.handle);
		return *this;
	}
	IObject* handle = nullptr;
};

struct WbemClassObject: Object<IWbemClassObject> {
	using Object<IWbemClassObject>::Object;

	template <typename T>
	T get(const std::string& name) const;

	template <>
	std::string get(const std::string& name) const {
		return converter.to_bytes(_get(name, CIM_STRING).bstrVal);
	}

	template <>
	uint16_t get(const std::string& name) const {
		return _get(name, CIM_UINT16).uiVal;
	}

	std::string getObjectText() {
		BSTR str = nullptr;
		HRESULT hres = handle->GetObjectText(0, &str);
		if (FAILED(hres)) {
			throw std::runtime_error("WbemClassObject::getObjectText failed");
		}
		return String(str);
	}

private:
	Variant _get(const std::string& name, CIMTYPE expected_type) const {
		Variant variant;
		CIMTYPE actual_type = CIM_ILLEGAL;
		HRESULT hres = handle->Get(converter.from_bytes(name).c_str(), 0, &variant, &actual_type, 0);
		if (FAILED(hres)) {
			throw std::runtime_error("WbemClassObject::Get failed");
		}
		if (actual_type != expected_type) {
			throw std::runtime_error("WbemClassObject::Get return data of unexpected type");
		}
		return variant;
	}
};

struct EnumWbemClassObject: Object<IEnumWbemClassObject> {
	using Object<IEnumWbemClassObject>::Object;

	std::vector<WbemClassObject> next(size_t count = 1) {
		ULONG returned = 0;
		std::vector<IWbemClassObject*> buffer(count, nullptr);
		HRESULT hres = handle->Next(WBEM_INFINITE, count, buffer.data(), &returned);
		std::vector<WbemClassObject> result(buffer.begin(), buffer.begin() + returned);
		if (FAILED(hres)) {
			throw std::runtime_error("EnumWbemClassObject::Next failed");
		}
		return result;
	}

	std::vector<WbemClassObject> getAll(size_t batch_size = 10) {
		std::vector<wmi::WbemClassObject> result;
		while (true) {
			std::vector<wmi::WbemClassObject> objects = next(batch_size);
			if (objects.size() == 0) {
				break;
			}
			for (auto& object: objects) {
				result.push_back(std::move(object));
			}
		}
		return result;
	}
};

struct WbemServices: Object<IWbemServices> {
	using Object<IWbemServices>::Object;

	void setProxyBlanket() {
		HRESULT hres = CoSetProxyBlanket(
			handle,
			RPC_C_AUTHN_WINNT,
			RPC_C_AUTHZ_NONE,
			NULL,
			RPC_C_AUTHN_LEVEL_CALL,
			RPC_C_IMP_LEVEL_IMPERSONATE,
			NULL,
			EOAC_NONE
		);

		if (FAILED(hres)) {
			throw std::runtime_error("CoSetProxyBlanket failed");
		}
	}

	EnumWbemClassObject execQuery(const std::string& query) const {
		IEnumWbemClassObject* enumerator = nullptr;
		HRESULT hres = handle->ExecQuery(
			bstr_t("WQL"),
			bstr_t(query.c_str()),
			WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
			NULL,
			&enumerator
		);

		if (FAILED(hres)) {
			throw std::runtime_error("IWbemServices::ExecQuery failed");
		}

		return enumerator;
	}
};

struct WbemLocator: Object<IWbemLocator> {
	WbemLocator() {
		HRESULT hres = CoCreateInstance(
			CLSID_WbemLocator,
			0,
			CLSCTX_INPROC_SERVER,
			IID_IWbemLocator, (LPVOID *)&handle);

		if (FAILED(hres)) {
			throw std::runtime_error("CoCreateInstance failed (WbemLocator)");
		}
	}

	WbemServices connectServer(const std::string& path) {
		IWbemServices* services = nullptr;

		HRESULT hres = handle->ConnectServer(
			_bstr_t(path.c_str()),
			NULL,
			NULL,
			_bstr_t("MS_409"),
			NULL,
			0,
			0,
			&services
		);

		if (FAILED(hres)) {
			throw std::runtime_error("IWbemLocator::ConnectServer failed");
		}

		return services;
	}
};

}
