
#include "Test.hpp"
#include "Action.hpp"

namespace IR {

std::string Test::name() const {
	return Id(ast_node->name, stack).value();
}

std::vector<std::string> Test::parent_names() const {
	std::vector<std::string> result;

	for (auto parent: ast_node->parents) {
		result.push_back(Id(parent, stack).value());
	}

	return result;
}

std::list<std::shared_ptr<IR::Test>> Test::get_test_path(const std::shared_ptr<IR::Test>& test) {
	std::list<std::shared_ptr<IR::Test>> result;

	/*
		For every parent. Get his parents. Iterate though them. If we already have a parent - don't include it
		Then do the same check for the parents themselves
	*/

	for (auto parent: test->parents) {
		concat_unique(result, get_test_path(parent));
	}

	concat_unique(result, {test});

	return result;
}

std::string Test::description() const {
	if (attrs.is_null()) {
		return "";
	}
	return attrs.value("description", "");
}

bool Test::snapshots_needed() const {
	if (attrs.is_null()) {
		return true;
	}
	return !attrs.value("no_snapshots", false);
}

std::set<std::shared_ptr<Controller>> Test::get_all_controllers() const {
	std::set<std::shared_ptr<Controller>> all_controllers;
	for (auto& machine: get_all_machines()) {
		all_controllers.insert(machine);
	}
	for (auto& flash_drive: get_all_flash_drives()) {
		all_controllers.insert(flash_drive);
	}
	return all_controllers;
}

std::set<std::shared_ptr<Machine>> Test::get_all_machines() const {
	std::set<std::shared_ptr<Machine>> result;
	for (auto& parent: parents) {
		for (auto& vm: parent->get_all_machines()) {
			result.insert(vm);
		}
	}

	for (auto& vm: mentioned_machines) {
		result.insert(vm);
	}

	return result;
}

std::set<std::shared_ptr<Network>> Test::get_all_networks() const {
	std::set<std::shared_ptr<Network>> result;
	for (auto& parent: parents) {
		for (auto& nw: parent->get_all_networks()) {
			result.insert(nw);
		}
	}

	for (auto& nw: mentioned_networks) {
		result.insert(nw);
	}

	return result;
}

std::set<std::shared_ptr<FlashDrive>> Test::get_all_flash_drives() const {
	std::set<std::shared_ptr<FlashDrive>> result;
	for (auto& parent: parents) {
		for (auto& fd: parent->get_all_flash_drives()) {
			result.insert(fd);
		}
	}

	for (auto& fd: mentioned_flash_drives) {
		result.insert(fd);
	}

	return result;
}

Test::CacheStatus Test::cache_status() {
	if (_cache_status == CacheStatus::Unknown) {
		if (is_cache_ok()) {
			_cache_status = CacheStatus::OK;
		} else if (is_cache_miss()) {
			_cache_status = CacheStatus::Miss;
		} else {
			_cache_status = CacheStatus::Empty;
		}
	}
	return _cache_status;
}

std::string to_string(Test::CacheStatus status) {
	switch (status) {
		case Test::CacheStatus::Unknown:
			return "unknown";
		case Test::CacheStatus::Empty:
			return "empty";
		case Test::CacheStatus::OK:
			return "ok";
		case Test::CacheStatus::Miss:
			return "miss";
		default:
			throw std::runtime_error("Invalid CacheStatus value");
	}
}

nlohmann::json Test::meta() {
	return {
		{"name", name()},
		{"parents", parent_names()},
		{"snapshots_needed", snapshots_needed()},
		{"description", description()},
		{"cksum", cksum},
		{"cache_status", to_string(cache_status())}
	};
}

void Test::report(const fs::path& report_folder_) {
	fs::path report_folder = report_folder_ / name();
	fs::path meta_file = report_folder / "meta.json";
	if (fs::exists(meta_file)) {
		std::ifstream file(meta_file);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file " + meta_file.generic_string());
		}
		nlohmann::json j;
		file >> j;
		if (j.at("cksum") != cksum) {
			throw std::runtime_error("Can't write the report into an existing report folder because \"" + name() + "\" test's checksum has been changed");
		}
	} else {
		fs::create_directories(report_folder);
		std::ofstream file(meta_file);
		file << meta().dump(2);
	}
}

bool Test::is_cache_ok() const {
	for (auto parent: parents) {
		if (parent->cache_status() != CacheStatus::OK) {
			return false;
		}
	}

	//check networks aditionally
	for (auto network: get_all_networks()) {
		if (network->is_defined() &&
			network->check_config_relevance())
		{
			continue;
		}
		return false;
	}

	for (auto controller: get_all_controllers()) {
		if (controller->is_defined() &&
			controller->has_snapshot("_init") &&
			controller->check_metadata_version() &&
			controller->check_config_relevance() &&
			controller->has_snapshot(name()) &&
			(controller->get_snapshot_cksum(name()) == cksum))
		{
			continue;
		}
		return false;
	}
	return true;
}

bool Test::is_cache_miss() const {
	for (auto parent: parents) {
		if (parent->cache_status() == CacheStatus::Miss) {
			return true;
		}
	}

	//check networks aditionally
	for (auto netc: get_all_networks()) {
		if (netc->is_defined()) {
			if (!netc->check_config_relevance()) {
				return true;
			}
		}
	}

	for (auto controller: get_all_controllers()) {
		if (controller->is_defined()) {
			if (controller->has_snapshot(name())) {
				if (controller->get_snapshot_cksum(name()) != cksum) {
					return true;
				}
				if (!controller->check_config_relevance()) {
					return true;
				}
			}
		}
	}

	return false;
}

std::set<std::string> TestRun::get_unsuccessful_parents_names() const {
	std::set<std::string> result;
	for (auto& parent: parents) {
		if (parent->exec_status == ExecStatus::Unknown) {
			throw std::runtime_error("get_unsuccessful_parents_names failed because parent " + parent->test->name() + " test's status is unknown");
		} else if (parent->exec_status != ExecStatus::Passed) {
			result.insert(parent->test->name());
		}
	}
	return result;
}

std::string to_string(TestRun::ExecStatus status) {
	switch (status) {
		case TestRun::ExecStatus::Unknown:
			return "unknown";
		case TestRun::ExecStatus::Passed:
			return "passed";
		case TestRun::ExecStatus::Failed:
			return "failed";
		case TestRun::ExecStatus::Skipped:
			return "skipped";
		default:
			throw std::runtime_error("Invalid ExecStatus value");
	}
}

std::string timepoint_to_str(std::chrono::system_clock::time_point tp) {
	auto t = std::chrono::system_clock::to_time_t(tp);
	std::stringstream ss;
	ss << std::put_time(std::localtime(&t), "%FT%T%z");
	return ss.str();
}

nlohmann::json TestRun::meta() {
	return {
		{"test_name", test->name()},
		{"exec_status", to_string(exec_status)},
		{"start_timestamp", timepoint_to_str(start_timestamp)},
		{"stop_timestamp", timepoint_to_str(stop_timestamp)}
	};
}

void TestRun::report_begin(const fs::path& report_folder) {
	fs::create_directories(report_folder / name);
	output_file = std::ofstream(report_folder / name / "log.txt");
}

void TestRun::report_screenshot(const fs::path& report_folder, const stb::Image<stb::RGB>& screenshot) {
	screenshot.write_png((report_folder / name / "screenshot.png").generic_string());
}

void TestRun::report_end(const fs::path& report_folder) {
	output_file.close();
	std::ofstream file(report_folder / name / "meta.json");
	file << meta().dump(2);
}

}
