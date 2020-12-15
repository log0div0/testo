
#pragma once

#include "WbemClassObject.hpp"
#include "EnumWbemClassObject.hpp"
#include <string>
#include <wbemcli.h>

namespace wmi {

struct Call;

struct WbemServices: Object<IWbemServices> {
	using Object<IWbemServices>::Object;

	void setProxyBlanket();

	EnumWbemClassObject execQuery(const std::string& query) const;
	WbemClassObject execMethod(const std::string& className, const std::string& methodName, WbemClassObject& input) const;
	WbemClassObject getObject(const std::string& name) const;

	Call call(const std::string& class_name, const std::string& method_name) const;
};

}
