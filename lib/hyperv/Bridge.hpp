
#pragma once

#include <wmi/WbemClassObject.hpp>
#include <wmi/WbemServices.hpp>

namespace hyperv {

struct Bridge {
	Bridge(
		wmi::WbemClassObject virtualEthernetSwitch_,
		wmi::WbemServices services_
	);

	std::string name() const;
	void destroy();

	wmi::WbemClassObject virtualEthernetSwitch;
	wmi::WbemServices services;
};

}
