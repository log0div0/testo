
#pragma once

#include <wmi/WbemClassObject.hpp>
#include <wmi/WbemServices.hpp>

namespace hyperv {

struct ResourceTemplate {
	ResourceTemplate(wmi::WbemServices services_, const std::string& type, const std::string& subtype);

	ResourceTemplate& put(const std::string& name, const wmi::Variant& value, CIMTYPE type = 0);
	wmi::WbemClassObject addTo(const wmi::WbemClassObject& systemSettings);

	wmi::WbemServices services;
	wmi::WbemClassObject settings;
};

}
