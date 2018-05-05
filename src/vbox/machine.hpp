
#pragma once

#include "api.hpp"
#include "storage_controller.hpp"
#include <vector>

namespace vbox {

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

	IMachine* handle = nullptr;
};

}
