
#pragma once

#include "Machine.hpp"
#include "Bridge.hpp"
#include <wmi/WbemLocator.hpp>

namespace hyperv {

enum class HardDiskType: uint16_t {
	Fixed = 2,
	Dynamic = 3,
	Differencing = 4
};

enum class HardDiskFormat: uint16_t {
	VHD = 2,
	VHDX = 3,
	VHDSet = 4
};

struct Connect {
	Connect();

	std::vector<Machine> machines() const;
	Machine machine(const std::string& name) const;
	Machine defineMachine(const std::string& name);
	std::vector<Bridge> bridges() const;
	Bridge bridge(const std::string& name) const;
	Bridge defineBridge(const std::string& name);
	std::string defaultVirtualHardDiskPath() const;
	void createDynamicHardDisk(const std::string& path, size_t size, HardDiskFormat format);
	void createDifferencingHardDisk(const std::string& path, const std::string& parent, HardDiskFormat format);

	wmi::WbemLocator locator;
	wmi::WbemServices services;
};

}
