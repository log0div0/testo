
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

	String compose_machine_filename(
		const Utf16String& name,
		const Utf16String& group,
		const Utf16String& create_flags,
		const Utf16String& base_folder
	);

	IVirtualBox* handle = nullptr;
};

}
