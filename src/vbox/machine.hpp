
#pragma once

#include "api.hpp"
#include "storage_controller.hpp"
#include "medium_attachment.hpp"
#include "medium.hpp"
#include "progress.hpp"
#include "safe_array.hpp"
#include <vector>
#include <ostream>

namespace vbox {

struct Session;

struct Machine {
	Machine() = default;
	Machine(IMachine* handle);
	~Machine();

	Machine(const Machine&) = delete;
	Machine& operator=(const Machine&) = delete;
	Machine(Machine&& other);
	Machine& operator=(Machine&& other);

	std::string name() const;
	void save_settings();
	std::string settings_file_path() const;

	std::vector<StorageController> storage_controllers() const;
	std::vector<MediumAttachment> medium_attachments() const;
	StorageController add_storage_controller(const std::string& name, StorageBus storage_bus);
	void attach_device_without_medium(const std::string& name, int controller_port, int device, DeviceType device_type);
	void attach_device(const std::string& name, int controller_port, int device, DeviceType device_type, const Medium& medium);
	SafeArray unregister(CleanupMode cleanup_mode);
	Progress delete_config(SafeArray mediums);
	Progress launch_vm_process(Session& session, const std::string& name, const std::string& environment = "");

	void vram_size(ULONG size);
	void memory_size(ULONG size);

	MachineState state() const;
	SessionState session_state() const;

	void lock_machine(Session& session, LockType lock_type);

	IMachine* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const Machine& machine);

}
