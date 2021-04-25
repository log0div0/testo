
#include "Test.hpp"
#include "Action.hpp"

namespace IR {

std::string Test::name() const {
	return StringTokenUnion(ast_node->name, stack).resolve();
}

std::vector<std::string> Test::parent_names() const {
	std::vector<std::string> result;

	for (auto parent: ast_node->parents) {
		result.push_back(StringTokenUnion(parent, stack).resolve());
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

}
