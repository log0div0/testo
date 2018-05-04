
#pragma once

#include "machine.hpp"
#include <vector>

namespace vbox {

struct VirtualBox {
	VirtualBox() = default;
	VirtualBox(IVirtualBox* handle);
	~VirtualBox();

	VirtualBox(const VirtualBox&) = delete;
	VirtualBox& operator=(const VirtualBox&) = delete;

	VirtualBox(VirtualBox&& other);
	VirtualBox& operator=(VirtualBox&& other);

	std::vector<Machine> machines() const;
	std::vector<String> machine_groups() const;

	IVirtualBox* handle = nullptr;
};

}
