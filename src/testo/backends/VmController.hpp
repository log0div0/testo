
#pragma once

#include "VM.hpp"

struct VmController {
	VmController() = delete;
	VmController(std::shared_ptr<VM> vm): vm(vm) {}

	void create_vm();
	void create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed);
	void restore_snapshot(const std::string& snapshot);
	void delete_snapshot_with_children(const std::string& snapshot);
	bool has_snapshot(const std::string& snapshot);
	std::string get_snapshot_cksum(const std::string& snapshot);

	bool has_key(const std::string& key);
	std::string get_metadata(const std::string& key);
	void set_metadata(const std::string& key, const std::string& value);

	std::shared_ptr<VM> vm;

private:
	void write_metadata_file(const fs::path& file, const nlohmann::json& metadata);
	nlohmann::json read_metadata_file(const fs::path& file) const;
};
