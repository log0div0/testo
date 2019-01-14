
#pragma once

#include "api.hpp"
#include "storage_controller.hpp"
#include "usb_controller.hpp"
#include "medium_attachment.hpp"
#include "network_adapter.hpp"
#include "medium.hpp"
#include "progress.hpp"
#include "safe_array.hpp"
#include "snapshot.hpp"
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
	StorageController storage_controller_by_name(const std::string& name) const;
	std::vector<USBController> usb_controllers() const;
	USBController usb_controller_by_name(const std::string& name) const;
	std::vector<MediumAttachment> medium_attachments() const;
	std::vector<MediumAttachment> medium_attachments_of_controller(const std::string& name) const;
	StorageController add_storage_controller(const std::string& name, StorageBus storage_bus);
	void mount_medium(const std::string& controller, int controller_port, int device, Medium& medium, bool force);
	void unmount_medium(const std::string& controller, int controller_port, int device, bool force);
	USBController add_usb_controller(const std::string& name, USBControllerType type);
	void attach_device_without_medium(const std::string& name, int controller_port, int device, DeviceType device_type);
	void attach_device(const std::string& name, int controller_port, int device, DeviceType device_type, const Medium& medium);
	void detach_device(const std::string& name, int controller_port, int device);

	std::vector<std::string> getExtraDataKeys() const;
	std::string getExtraData(const std::string& key) const;
	void setExtraData(const std::string& key, const std::string& value) const;

	NetworkAdapter getNetworkAdapter(ULONG slot) const;

	bool hasSnapshot(const std::string& name) const;
	Snapshot findSnapshot(const std::string& name) const;
	Progress takeSnapshot(const std::string& name, const std::string& description = "", bool pause = false);
	Progress restoreSnapshot(Snapshot& snapshot);
	Progress deleteSnapshot(Snapshot& snapshot);

	Progress saveState() const;

	SafeArray unregister(CleanupMode cleanup_mode);
	Progress delete_config(SafeArray mediums);
	Progress launch_vm_process(Session& session, const std::string& name, const std::string& environment = "");

	void vram_size(ULONG size);
	void memory_size(ULONG size);
	void cpus(ULONG num);

	MachineState state() const;
	SessionState session_state() const;

	void lock_machine(Session& session, LockType lock_type);

	IMachine* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const Machine& machine);

}
