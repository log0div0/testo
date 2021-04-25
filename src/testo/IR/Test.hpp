
#pragma once

#include "Machine.hpp"
#include "FlashDrive.hpp"
#include "Network.hpp"
#include <fstream>
#include <sstream>

namespace IR {

struct Test: Object<AST::Test> {

	enum class CacheStatus {
		Unknown,
		Empty,
		OK,
		Miss
	};

	static std::list<std::shared_ptr<Test>> get_test_path(const std::shared_ptr<Test>& test);

	static std::string type_name() { return "test"; }
	std::string name() const;
	std::vector<std::string> parent_names() const;

	std::set<std::shared_ptr<Test>> parents;

	std::set<std::shared_ptr<Controller>> get_all_controllers() const;
	std::set<std::shared_ptr<Network>> get_all_networks() const;
	std::set<std::shared_ptr<Machine>> get_all_machines() const;
	std::set<std::shared_ptr<FlashDrive>> get_all_flash_drives() const;

	std::string description() const;
	bool snapshots_needed() const;

	std::stringstream cksum_input;
	std::string cksum;

	CacheStatus cache_status();

	std::chrono::system_clock::time_point start_timestamp;
	std::chrono::system_clock::time_point stop_timestamp;
	std::ofstream output_file;
	nlohmann::json attrs;

	std::set<std::shared_ptr<Machine>> mentioned_machines;
	std::set<std::shared_ptr<Network>> mentioned_networks;
	std::set<std::shared_ptr<FlashDrive>> mentioned_flash_drives;

private:
	CacheStatus _cache_status = CacheStatus::Unknown;
	bool is_cache_ok() const;
	bool is_cache_miss() const;
};

}
