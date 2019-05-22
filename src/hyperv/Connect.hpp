
#pragma once

#include "Machine.hpp"

namespace hyperv {

struct Connect {
	Connect();

	std::vector<Machine> machines() const;
	Machine machine(const std::string& name) const;

	wmi::WbemLocator locator;
	wmi::WbemServices services;
};

}
