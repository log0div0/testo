
#pragma once

#include "Display.hpp"

namespace hyperv {

struct Machine {
	Machine(wmi::WbemClassObject computerSystem_, wmi::WbemServices services_);

	std::string name() const;
	bool is_running() const;
	Display display() const;

	wmi::WbemClassObject computerSystem;
	wmi::WbemServices services;
};

}
