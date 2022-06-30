
#pragma once

#include "Machine.hpp"
#include "FlashDrive.hpp"
#include "Network.hpp"

namespace IR {

struct TestRun;

struct Test: Object<AST::Test> {

	enum class CacheStatus {
		Unknown,
		Empty,
		OK,
		Miss
	};

	enum class SnapshotPolicy {
		Unknown,
		Always,
		Never,
		Auto
	};

	static std::list<std::shared_ptr<Test>> get_test_path(const std::shared_ptr<Test>& test);

	fs::path get_source_file_path() const;

	static std::string type_name() { return "test"; }
	std::string name() const;
	std::vector<std::string> parent_names() const;

	// both in a fixed order to make test planning stable
	std::vector<std::shared_ptr<Test>> parents;
	std::vector<std::weak_ptr<Test>> children;

	std::list<std::shared_ptr<Test>> get_children() const;

	std::set<std::shared_ptr<Controller>> get_all_controllers() const;
	std::set<std::shared_ptr<Network>> get_all_networks() const;
	std::set<std::shared_ptr<Machine>> get_all_machines() const;
	std::set<std::shared_ptr<FlashDrive>> get_all_flash_drives() const;

	std::string title() const;
	std::string description() const;
	std::vector<std::string> depends_on() const;
	const std::set<std::string>& get_all_test_names_in_subtree();
	const std::set<std::string>& get_external_dependencies();

	void add_snapshot_ref(const TestRun* test_run);
	void remove_snapshot_ref(const TestRun* test_run);
	bool can_delete_hypervisor_snaphots() const;
	bool is_hypervisor_snapshot_needed() const;
	bool has_hypervisor_snapshot() const;
	SnapshotPolicy snapshot_policy() const;

	std::stringstream cksum_input;
	std::string cksum;

	CacheStatus cache_status() const;
	bool is_up_to_date() const {
		return cache_status() == IR::Test::CacheStatus::OK;
	}
	bool all_parents_are_up_to_date() const;

	const nlohmann::json& attrs() const;

	std::set<std::shared_ptr<Machine>> mentioned_machines;
	std::set<std::shared_ptr<Network>> mentioned_networks;
	std::set<std::shared_ptr<FlashDrive>> mentioned_flash_drives;

	bool has_repls = false;

private:
	std::set<const TestRun*> snapshot_refs;
	mutable CacheStatus _cache_status = CacheStatus::Unknown;
	mutable SnapshotPolicy _snapshot_policy = SnapshotPolicy::Unknown;
	bool is_cache_ok() const;
	bool is_cache_miss() const;
	std::unique_ptr<std::set<std::string>> all_test_names_in_subtree;
	std::unique_ptr<std::set<std::string>> external_dependencies;
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
	std::set<std::string> unsuccessful_deps_names;
	std::set<std::string> found_bugs;

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
