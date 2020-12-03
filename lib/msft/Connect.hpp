
#pragma once

#include "Disk.hpp"

namespace msft {

struct Connect {
	Connect();

	std::vector<Disk> disks() const;
	Disk virtualDisk(const std::string& location) const;

	wmi::WbemLocator locator;
	wmi::WbemServices services;
};

}
