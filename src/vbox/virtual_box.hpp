
#pragma once

#include "machine.hpp"
#include <vector>

namespace vbox {

struct VirtualBox {
	VirtualBox(IVirtualBox* handle);
	~VirtualBox();

	VirtualBox(const VirtualBox&) = delete;
	VirtualBox& operator=(const VirtualBox&) = delete;

	VirtualBox(VirtualBox&& other);
	VirtualBox& operator=(VirtualBox&& other);

	std::vector<Machine> machines() const;
	std::vector<Medium> dvd_images() const;
	std::vector<Medium> hard_disks() const;
	Machine find_machine(const std::string& name) const;
	std::vector<std::string> machine_groups() const;

	std::string compose_machine_filename(
		const std::string& name,
		const std::string& group,
		const std::string& create_flags,
		const std::string& base_folder
	) const;

	Machine create_machine(
		const std::string& settings_file,
		const std::string& name,
		const std::vector<std::string>& groups,
		const std::string& os_type_id,
		const std::string& flags
	);

	Medium create_medium(
		const std::string& format,
		const std::string& location,
		AccessMode access_mode,
		DeviceType device_type
	);

	Medium open_medium(
		const std::string& location,
		DeviceType device_type,
		AccessMode access_mode,
		bool force_new_uuid
	);

	void register_machine(const Machine& machine);

	IVirtualBox* handle = nullptr;
};

}
