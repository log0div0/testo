
#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>

#include <Wbemidl.h>
#include <comdef.h>
#include <propvarutil.h>

namespace wmi {

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
	Variant(bool value) {
		InitVariantFromBoolean(value, this);
	}
	Variant(const char* str) {
		InitVariantFromString(bstr_t(str), this);
	}
	Variant(const std::string& str) {
		InitVariantFromString(bstr_t(str.c_str()), this);
	}
	Variant(const std::vector<uint8_t>& buffer) {
		InitVariantFromBuffer(buffer.data(), buffer.size(), this);
	}
	Variant(const std::vector<std::string>& strs) {
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
	Variant(uint16_t value) {
		InitVariantFromUInt16(value, this);
	}
	Variant(int16_t value) {
		InitVariantFromInt16(value, this);
	}
	Variant(uint32_t value) {
		InitVariantFromUInt32(value, this);
	}
	Variant(int32_t value) {
		InitVariantFromInt32(value, this);
	}
	Variant(uint64_t value) {
		InitVariantFromUInt64(value, this);
	}
	Variant(int64_t value) {
		InitVariantFromInt64(value, this);
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

	operator std::string() const {
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

	operator std::vector<std::string>() const
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

	operator uint16_t() const {
		try {
			check_type(VT_UI2);
			return uiVal;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	operator int16_t() const {
		try {
			check_type(VT_I2);
			return iVal;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	operator uint32_t() const {
		try {
			check_type(VT_UI4);
			return ulVal;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	operator int32_t() const {
		try {
			check_type(VT_I4);
			return lVal;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	operator std::vector<uint8_t>() const {
		try {
			check_type(VARENUM(VT_ARRAY | VT_UI1));
			std::vector<uint8_t> result(VariantGetElementCount(*this));
			throw_if_failed(VariantToBuffer(*this, result.data(), result.size()));
			return result;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	template <typename T>
	T get() const {
		return *this;
	}

private:
	void check_type(VARENUM expected) const {
		if (vt != expected) {
			throw std::runtime_error("Expected type = " + to_hex(expected) + " , actual type = " + to_hex(vt));
		}
	}
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
			throw_if_failed(handle->Get(bstr_t(name.c_str()), 0, &variant, nullptr, 0));
			return variant;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	std::string getObjectText() {
		try {
			BSTR str = nullptr;
			throw_if_failed(handle->GetObjectText(0, &str));
			return (const char*)bstr_t(str, false);
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
			if (object) {
				return WbemClassObject(object);
			} else {
				return WbemClassObject();
			}
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

	WbemClassObject& put(const std::string& name, const Variant& value, CIMTYPE type = 0) {
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

	WbemClassObject clone() {
		try {
			IWbemClassObject* copy = nullptr;
			throw_if_failed(handle->Clone(&copy));
			return copy;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
};

struct WbemObjectTextSrc: Object<IWbemObjectTextSrc> {
	WbemObjectTextSrc() {
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

	std::string getText(const WbemClassObject& object, ULONG format = WMI_OBJ_TEXT_WMI_DTD_2_0) {
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
			if (objects.size() == 0) {
				throw std::runtime_error("No object found");
			}
			if (objects.size() != 1) {
				throw std::runtime_error("Found more objects than expected");
			}
			return std::move(objects.at(0));
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
};

struct Call;

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

	Call call(const std::string& class_name, const std::string& method_name) const;

	wmi::WbemClassObject getResourceTemplate(const std::string& type, const std::string& subtype) {
		try {
			return execQuery(
				"SELECT * FROM " + type + " "
				"WHERE InstanceID LIKE \"%Default\" "
				"AND ResourceSubType=\"" + subtype + "\""
			).getOne().clone();
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	wmi::WbemClassObject addResource(const wmi::WbemClassObject& target, const wmi::WbemClassObject& resourceTemplate);
	wmi::WbemClassObject modifyResource(const wmi::WbemClassObject& resource);
};

struct Call {
	Call(WbemServices services_, std::string class_name_, std::string method_name_):
		services(std::move(services_)), class_name(std::move(class_name_)), method_name(std::move(method_name_))
	{
		try {
			auto method = services.getObject(class_name).getMethod(method_name);
			if (method.handle) {
				method_instance = method.spawnInstance();
			}
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	Call& with(std::string name, const Variant& value) {
		try {
			method_instance.put(name, value);
			return *this;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	Call& with(std::string name, const WbemClassObject& object) {
		try {
			return with(name, WbemObjectTextSrc().getText(object));
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	Call& with(std::string name, const std::vector<WbemClassObject>& objects) {
		try {
			std::vector<std::string> strings;
			WbemObjectTextSrc objectTextSrc;
			for (auto& object: objects) {
				strings.push_back(objectTextSrc.getText(object));
			}
			return with(name, strings);
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	WbemClassObject exec(WbemClassObject object) {
		try {
			auto result = services.execMethod(object.path(), method_name, method_instance);
			int32_t returnValue = result.get("ReturnValue");
			if (returnValue != 0) {
				if (returnValue == 4096) {
					std::string jobRef = result.get("Job");
					while (true) {
						auto job = services.getObject(jobRef);
						int32_t jobState = job.get("JobState");
						if (jobState == 4) {
							std::this_thread::sleep_for(std::chrono::milliseconds(100));
							continue;
						}
						if (jobState == 7) {
							break;
						}
						std::string errorDescription = job.get("ErrorDescription");
						throw std::runtime_error(errorDescription);
					}
				} else {
					throw std::runtime_error("ReturnValue == " + std::to_string(returnValue));
				}
			}
			return result;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}
	WbemClassObject exec() {
		try {
			return exec(services.execQuery("SELECT * FROM " + class_name).getOne());
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

private:
	WbemServices services;
	std::string class_name;
	std::string method_name;
	WbemClassObject method_instance;
};

inline Call WbemServices::call(const std::string& class_name, const std::string& method_name) const {
	return Call(*this, class_name, method_name);
}

inline wmi::WbemClassObject WbemServices::addResource(const wmi::WbemClassObject& target, const wmi::WbemClassObject& resourceTemplate) {
	try {
		auto result = call("Msvm_VirtualSystemManagementService", "AddResourceSettings")
				.with("AffectedConfiguration", target.path())
				.with("ResourceSettings", std::vector<wmi::WbemClassObject>{resourceTemplate})
				.exec();
		std::vector<std::string> refs = result.get("ResultingResourceSettings");
		return getObject(refs.at(0));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

inline wmi::WbemClassObject WbemServices::modifyResource(const wmi::WbemClassObject& resource) {
	try {
		auto result = call("Msvm_VirtualSystemManagementService", "ModifyResourceSettings")
				.with("ResourceSettings", std::vector<wmi::WbemClassObject>{resource})
				.exec();
		std::vector<std::string> refs = result.get("ResultingResourceSettings");
		return getObject(refs.at(0));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

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