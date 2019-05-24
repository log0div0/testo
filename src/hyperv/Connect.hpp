
#pragma once

#include "Machine.hpp"
#include "Bridge.hpp"

namespace hyperv {

struct Connect {
	Connect();

	std::vector<Machine> machines() const;
	Machine machine(const std::string& name) const;
	Machine defineMachine(const std::string& name);
	std::vector<Bridge> bridges() const;
	Bridge bridge(const std::string& name) const;
	Bridge defineBridge(const std::string& name);

	wmi::WbemLocator locator;
	wmi::WbemServices services;
};

}
