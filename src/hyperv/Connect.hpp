
#pragma once

#include "Machine.hpp"

namespace hyperv {

struct Connect {
	Connect();

	std::vector<Machine> machines() const;

	wmi::WbemLocator locator;
	wmi::WbemServices services;
};

}
