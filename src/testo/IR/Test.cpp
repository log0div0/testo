
#include "Test.hpp"
#include "Action.hpp"
#include "Program.hpp"

#include <iomanip>

namespace IR {

std::string Test::name() const {
	return Id(ast_node->name, stack).value();
}

std::vector<std::string> Test::parent_names() const {
	std::vector<std::string> result;

	for (auto parent: ast_node->parents->all()) {
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

fs::path Test::get_source_file_path() const {
	return ast_node->begin().file;
}


const nlohmann::json& Test::attrs() const {
	if (_attrs.is_null()) {
		if (ast_node->attrs) {
			_attrs = IR::AttrBlock(ast_node->attrs, stack).to_json();
		} else {
			_attrs = nlohmann::json::object();
		}
	}
	return _attrs;
}

std::string Test::title() const {
	return attrs().value("title", "");
}

std::string Test::description() const {
	return attrs().value("description", "");
}

std::vector<std::string> Test::depends_on() const {
	return attrs().value("depends_on", std::vector<std::string>());
}

const std::set<std::string>& Test::get_all_test_names_in_subtree() {
	if (!all_test_names_in_subtree) {
		all_test_names_in_subtree = std::make_unique<std::set<std::string>>(_get_all_test_names_in_subtree());
	}
	return *all_test_names_in_subtree;

}

const std::set<std::string>& Test::get_external_dependencies() {
	if (!external_dependencies) {
		external_dependencies = std::make_unique<std::set<std::string>>(_get_external_dependencies());
	}
	return *external_dependencies;

}

std::set<std::string> Test::_get_all_test_names_in_subtree() {
	std::set<std::string> result = {name()};

	for (auto& child: children) {
		std::set<std::string> names = child.lock()->_get_all_test_names_in_subtree();
		result.insert(names.begin(), names.end());
	}

	return result;
}

std::set<std::string> Test::_get_external_dependencies() {
	std::set<std::string> result;

	// push my deps in the list
	std::vector<std::string> my_deps = depends_on();
	result.insert(my_deps.begin(), my_deps.end());

	// push child deps in the list
	for (auto& child: children) {
		std::set<std::string> child_deps = child.lock()->get_external_dependencies();
		result.insert(child_deps.begin(), child_deps.end());
	}

	// early exit
	if (!result.size()) {
		return result;
	}

	std::set<std::string> all_names = get_all_test_names_in_subtree();

	// my deps validation
	for (auto& dep: my_deps) {
		if (dep == name()) {
			throw std::runtime_error("Test '" + name() + "' can't depend on itself");
		} else {
			if (all_names.count(dep)) {
				throw std::runtime_error("Test '" + name() + "' can't depend on its child '" + dep + "'");
			}
		}
	}

	// remove internal deps
	for (auto& name: all_names) {
		result.erase(name);
	}

	return result;
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

Test::CacheStatus Test::cache_status() const {
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

void Test::add_snapshot_ref(const TestRun* test_run) {
	snapshot_refs.insert(test_run);
}
void Test::remove_snapshot_ref(const TestRun* test_run) {
	snapshot_refs.erase(test_run);
}

bool Test::can_delete_hypervisor_snaphots() const {
	return (snapshot_refs.size() == 0) && (snapshot_policy() == SnapshotPolicy::Auto);
}

bool Test::is_hypervisor_snapshot_needed() const {
	if (snapshot_policy() == SnapshotPolicy::Always) {
		return true;
	}
	return (snapshot_refs.size() > 0) && (snapshot_policy() == SnapshotPolicy::Auto);
}

Test::SnapshotPolicy Test::snapshot_policy() const {
	if (_snapshot_policy == SnapshotPolicy::Unknown) {
		if (attrs().count("snapshots") && attrs().count("no_snapshots")) {
			throw std::runtime_error("You can't use both 'snapshots' and 'no_snapshots' attributes at the same time. 'no_snapshots' is deprecated so use 'snapshots' instead");
		}
		if (attrs().count("no_snapshots")) {
			bool no_snapshots = attrs().at("no_snapshots");
			if (no_snapshots) {
				_snapshot_policy = SnapshotPolicy::Never;
			} else {
				_snapshot_policy = SnapshotPolicy::Always;
			}
		} else {
			std::string str = attrs().value("snapshots", IR::program->resolve_top_level_param("TESTO_SNAPSHOT_DEFAULT_POLICY"));
			if (str == "always") {
				_snapshot_policy = SnapshotPolicy::Always;
			} else if (str == "auto") {
				_snapshot_policy = SnapshotPolicy::Auto;
			} else if (str == "never") {
				_snapshot_policy = SnapshotPolicy::Never;
			} else {
				throw std::runtime_error("Unknown 'snapshot' attribute value: " + str);
			}
		}
	}
	return _snapshot_policy;
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

}
