
#pragma once

#include "api.hpp"
#include "string.hpp"
#include <ostream>

namespace vbox {

struct StorageController {
	StorageController(IStorageController* handle);
	~StorageController();

	StorageController(const StorageController&) = delete;
	StorageController& operator=(const StorageController&) = delete;

	StorageController(StorageController&& other);
	StorageController& operator=(StorageController&& other);

	std::string name() const;
	StorageBus bus() const;
	StorageControllerType controller_type() const;
	size_t port_count() const;
	bool host_io_cache() const;
	bool bootable() const;

	IStorageController* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const StorageController& storage_controller);

}
