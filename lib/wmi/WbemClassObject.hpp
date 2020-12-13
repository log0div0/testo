
#pragma once

#include "Object.hpp"
#include "Variant.hpp"
#include <wbemcli.h>

namespace wmi {

struct WbemClassObject: Object<IWbemClassObject> {
	using Object<IWbemClassObject>::Object;

	Variant get(const std::string& name) const;
	std::string getObjectText();
	std::string relpath() const;
	std::string path() const;
	WbemClassObject getMethod(const std::string& name) const;
	WbemClassObject spawnInstance() const;
	WbemClassObject& put(const std::string& name, const Variant& value, CIMTYPE type = 0);
	WbemClassObject clone();
};

}
