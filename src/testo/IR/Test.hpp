
#pragma once

#include "Machine.hpp"
#include "FlashDrive.hpp"
#include "Network.hpp"

namespace IR {

struct Test: Object<AST::Test> {

	enum class CacheStatus {
		Unknown,
		Empty,
		OK,
		Miss
	};

	static std::list<std::shared_ptr<Test>> get_test_path(const std::shared_ptr<Test>& test);

	fs::path get_source_file_path() const;

	static std::string type_name() { return "test"; }
	std::string name() const;
	std::vector<std::string> parent_names() const;

	// both in a fixed order to make test planning stable
	std::vector<std::shared_ptr<Test>> parents;
	std::vector<std::weak_ptr<Test>> children;

	std::set<std::shared_ptr<Controller>> get_all_controllers() const;
	std::set<std::shared_ptr<Network>> get_all_networks() const;
	std::set<std::shared_ptr<Machine>> get_all_machines() const;
	std::set<std::shared_ptr<FlashDrive>> get_all_flash_drives() const;

	std::string title() const;
	std::string description() const;
	std::vector<std::string> depends_on() const;
	const std::set<std::string>& get_all_test_names_in_subtree();
	const std::set<std::string>& get_external_dependencies();

	bool snapshots_needed() const;

	std::stringstream cksum_input;
	std::string cksum;

	CacheStatus cache_status();
	bool is_up_to_date() {
		return cache_status() == IR::Test::CacheStatus::OK;
	}

	const nlohmann::json& attrs() const;

	std::set<std::shared_ptr<Machine>> mentioned_machines;
	std::set<std::shared_ptr<Network>> mentioned_networks;
	std::set<std::shared_ptr<FlashDrive>> mentioned_flash_drives;

private:
	CacheStatus _cache_status = CacheStatus::Unknown;
	bool is_cache_ok() const;
	bool is_cache_miss() const;
	std::optional<std::set<std::string>> all_test_names_in_subtree;
	std::optional<std::set<std::string>> external_dependencies;
	std::set<std::string> _get_all_test_names_in_subtree();
	std::set<std::string> _get_external_dependencies();
	mutable nlohmann::json _attrs;
};

struct TestRun {
	enum class ExecStatus {
		Unknown,
		Passed,
		Failed,
		Skipped
	};
	ExecStatus exec_status = ExecStatus::Unknown;

	std::string id = generate_uuid_v4();
	std::shared_ptr<Test> test;

	std::set<std::shared_ptr<TestRun>> parents;
	std::set<std::string> get_unsuccessful_parents_names() const;

	std::chrono::system_clock::time_point start_timestamp;
	std::chrono::system_clock::time_point stop_timestamp;
	auto duration() const {
		return stop_timestamp - start_timestamp;
	}

	std::string failure_message;
	std::string failure_stacktrace;
	std::string failure_category;
};

}
