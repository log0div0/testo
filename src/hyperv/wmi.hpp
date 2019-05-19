
#pragma once

#include <vector>
#include <string>
#include <locale>
#include <codecvt>
#include <sstream>

#include <Wbemidl.h>
#include <comdef.h>
#include <propvarutil.h>

namespace wmi {

extern std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;

template <typename T>
std::string to_hex(T t) {
	std::stringstream stream;
	stream << std::hex << "0x" << t;
	return stream.str();
}

struct ComError: std::runtime_error {
	ComError(HRESULT hresult_):
		std::runtime_error("HRESULT = " + to_hex(hresult_)),
		hresult(hresult_)
	{

	}

	HRESULT hresult;
};

inline void throw_if_failed(HRESULT hr) {
	if (FAILED(hr)) {
		throw ComError(hr);
	}
}

struct CoInitializer {
	CoInitializer() {
		try {
			throw_if_failed(CoInitializeEx(0, COINIT_MULTITHREADED));
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	void initalize_security() {
		try {
			throw_if_failed(CoInitializeSecurity(
				nullptr,
				-1,
				nullptr,
				nullptr,
				RPC_C_AUTHN_LEVEL_DEFAULT,
				RPC_C_IMP_LEVEL_IMPERSONATE,
				nullptr,
				EOAC_NONE,
				NULL
			));
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
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
	Variant(const std::string& str) {
		InitVariantFromString(bstr_t(str.c_str()), this);
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

	template <typename T>
	operator T() const {
		return get<T>();
	}

	template <typename T>
	T get() const;

	template <>
	std::string get() const {
		try {
			check_type(VT_BSTR);
			if (bstrVal == nullptr) {
				throw std::runtime_error("nullptr");
			}
			return converter.to_bytes(bstrVal);
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	template <>
	uint16_t get() const {
		try {
			check_type(VT_UI2);
			return uiVal;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	template <>
	int16_t get() const {
		try {
			check_type(VT_I2);
			return iVal;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	template <>
	uint32_t get() const {
		try {
			check_type(VT_UI4);
			return ulVal;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	template <>
	int32_t get() const {
		try {
			check_type(VT_I4);
			return lVal;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	template <>
	std::vector<uint8_t> get() const {
		try {
			check_type(VARENUM(VT_ARRAY | VT_UI1));
			long lowerBound = 0;
			long upperBound = 0;
			SafeArrayGetLBound(parray, 1 , &lowerBound);
			SafeArrayGetUBound(parray, 1, &upperBound);
			uint8_t* begin = nullptr;
			throw_if_failed(SafeArrayAccessData(parray, (void**)&begin));
			uint8_t* end = begin + (upperBound - lowerBound + 1);
			std::vector<uint8_t> result(begin, end);
			SafeArrayUnaccessData(parray);
			return result;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

private:
	void check_type(VARENUM expected) const {
		if (vt != expected) {
			throw std::runtime_error("Expected type = " + to_hex(expected) + " , actual type = " + to_hex(vt));
		}
	}
};

struct String {
	String(BSTR handle_): handle(handle_) {
		try {
			if (handle == nullptr) {
				throw std::runtime_error("nullptr");
			}
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	~String() {
		if (handle) {
			SysFreeString(handle);
			handle = nullptr;
		}
	}
	operator std::string() const {
		try {
			return converter.to_bytes(handle);
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	BSTR handle = nullptr;
};

template <typename IObject>
struct Object {
	Object() {}
	Object(IObject* handle_): handle(handle_) {
		try {
			if (handle == nullptr) {
				throw std::runtime_error("nullptr");
			}
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	~Object() {
		if (handle) {
			handle->Release();
			handle = nullptr;
		}
	}

	Object(const Object& other): handle(other.handle) {
		handle->AddRef();
	}
	Object& operator=(const Object& other) {
		if (handle) {
			handle->Release();
			handle = nullptr;
		}
		handle = other->handle;
		handle->AddRef();
		return *this;
	}
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

	Variant get(const std::string& name) const {
		try {
			Variant variant;
			throw_if_failed(handle->Get(converter.from_bytes(name).c_str(), 0, &variant, nullptr, 0));
			return variant;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	std::string getObjectText() {
		try {
			BSTR str = nullptr;
			throw_if_failed(handle->GetObjectText(0, &str));
			return String(str);
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	std::string relpath() const {
		try {
			return get("__RELPATH");
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	std::string path() const {
		try {
			return get("__PATH");
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	WbemClassObject getMethod(const std::string& name) const {
		try {
			IWbemClassObject* object = nullptr;
			throw_if_failed(handle->GetMethod(
				bstr_t(name.c_str()),
				0,
				&object,
				nullptr
			));
			return object;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	WbemClassObject spawnInstance() const {
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

	void put(const std::string& name, Variant value) {
		try {
			throw_if_failed(handle->Put(
				bstr_t(name.c_str()),
				0,
				&value,
				0
			));
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
};

struct EnumWbemClassObject: Object<IEnumWbemClassObject> {
	using Object<IEnumWbemClassObject>::Object;

	std::vector<WbemClassObject> next(size_t count) {
		try {
			ULONG returned = 0;
			std::vector<IWbemClassObject*> buffer(count, nullptr);
			HRESULT hres = handle->Next(WBEM_INFINITE, count, buffer.data(), &returned);
			std::vector<WbemClassObject> result(buffer.begin(), buffer.begin() + returned);
			throw_if_failed(hres);
			return result;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	std::vector<WbemClassObject> getAll(size_t batch_size = 1) {
		try {
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
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	WbemClassObject getOne() {
		try {
			auto objects = getAll();
			if (objects.size() != 1) {
				throw std::runtime_error("Get more object than expected");
			}
			return std::move(objects.at(0));
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
};

struct WbemServices: Object<IWbemServices> {
	using Object<IWbemServices>::Object;

	void setProxyBlanket() {
		try {
			throw_if_failed(CoSetProxyBlanket(
				handle,
				RPC_C_AUTHN_WINNT,
				RPC_C_AUTHZ_NONE,
				nullptr,
				RPC_C_AUTHN_LEVEL_CALL,
				RPC_C_IMP_LEVEL_IMPERSONATE,
				nullptr,
				EOAC_NONE
			));
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	EnumWbemClassObject execQuery(const std::string& query) const {
		try {
			IEnumWbemClassObject* enumerator = nullptr;
			throw_if_failed(handle->ExecQuery(
				L"WQL",
				bstr_t(query.c_str()),
				WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
				nullptr,
				&enumerator
			));
			return enumerator;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	WbemClassObject execMethod(const std::string& className, const std::string& methodName, WbemClassObject& input) const {
		try {
			IWbemClassObject* output = nullptr;
			throw_if_failed(handle->ExecMethod(
				bstr_t(className.c_str()),
				bstr_t(methodName.c_str()),
				0,
				nullptr,
				input.handle,
				&output,
				nullptr
			));
			return output;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	WbemClassObject getObject(const std::string& name) const {
		try {
			IWbemClassObject* object = nullptr;
			throw_if_failed(handle->GetObject(
				bstr_t(name.c_str()),
				0,
				nullptr,
				&object,
				nullptr
			));
			return object;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
};

struct WbemLocator: Object<IWbemLocator> {
	WbemLocator() {
		try {
			throw_if_failed(CoCreateInstance(
				CLSID_WbemLocator,
				0,
				CLSCTX_INPROC_SERVER,
				IID_IWbemLocator, (LPVOID *)&handle));
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	WbemServices connectServer(const std::string& path) {
		try {
			IWbemServices* services = nullptr;

			throw_if_failed(handle->ConnectServer(
				bstr_t(path.c_str()),
				nullptr,
				nullptr,
				L"MS_409",
				0,
				0,
				0,
				&services
			));

			return services;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
};

}
