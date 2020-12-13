
#pragma once

#include <wmi/WbemClassObject.hpp>
#include <wmi/WbemServices.hpp>

namespace msft {

struct Volume {
	Volume(wmi::WbemClassObject volume_, wmi::WbemServices services_);

	void format(const std::string& filesystem, const std::string& filesystemLabel);

	wmi::WbemClassObject volume;
	wmi::WbemServices services;
};

}
