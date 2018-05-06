
#pragma once

#include "api.hpp"
#include "storage_controller.hpp"
#include "medium_attachment.hpp"
#include "medium.hpp"
#include "progress.hpp"
#include <vector>
#include <ostream>

namespace vbox {

struct Session;

struct Machine {
	Machine(IMachine* handle);
	~Machine();

	Machine(const Machine&) = delete;
	Machine& operator=(const Machine&) = delete;
	Machine(Machine&& other);
	Machine& operator=(Machine&& other);

	std::string name() const;
	void save_settings();

	std::vector<StorageController> storage_controllers() const;
	std::vector<MediumAttachment> medium_attachments() const;
	StorageController add_storage_controller(const std::string& name, StorageBus storage_bus);
	std::vector<Medium> unregister(CleanupMode cleanup_mode);
	Progress delete_config(std::vector<Medium> mediums);

	void lock_machine(Session& session, LockType lock_type);

	IMachine* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const Machine& machine);

}
