
#include "WbemServices.hpp"
#include "Error.hpp"
#include "Call.hpp"
#include <comutil.h>

namespace wmi {

void WbemServices::setProxyBlanket() {
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

EnumWbemClassObject WbemServices::execQuery(const std::string& query) const {
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

WbemClassObject WbemServices::execMethod(const std::string& className, const std::string& methodName, WbemClassObject& input) const {
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

WbemClassObject WbemServices::getObject(const std::string& name) const {
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

Call WbemServices::call(const std::string& class_name, const std::string& method_name) const {
	return Call(*this, class_name, method_name);
}

}
