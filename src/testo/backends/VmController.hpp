
#pragma once

#include "VM.hpp"

struct VmController {
	VmController() = delete;
	VmController(std::shared_ptr<VM> vm): vm(vm) {}

	void create_vm();

	std::string get_metadata(const std::string& key);
	void set_metadata(const std::string& key, const std::string& value);

	std::shared_ptr<VM> vm;

private:
	void write_metadata_file(const fs::path& file, const nlohmann::json& metadata);
	nlohmann::json read_metadata_file(const fs::path& file) const;
};
