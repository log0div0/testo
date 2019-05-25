
#pragma once

#include "wmi.hpp"

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
