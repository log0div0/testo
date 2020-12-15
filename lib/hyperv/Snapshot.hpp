
#pragma once

#include <wmi/WbemClassObject.hpp>
#include <wmi/WbemServices.hpp>

namespace hyperv {

struct Snapshot {
	Snapshot(wmi::WbemClassObject virtualSystemSettingData_,
		wmi::WbemServices services_);

	std::string name() const;
	void setName(const std::string& name);

	void apply();
	void destroy();

	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
