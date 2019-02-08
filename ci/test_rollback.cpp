
#include "fakeit.hpp"
#include "catch2/catch.hpp"
#include "testo/Interpreter.hpp"
#include <iostream>
#include <algorithm>

using namespace fakeit;

nlohmann::json default_original_config = {
		{"cpus", 1},
		{"os_type", "ubuntu_64"},
		{"ram", 512},
		{"iso", "/ubuntu-16.04.5-server-amd64.iso"},
		{"disk_size", 4096},
		{"nic", {
			{
				{"name", "nat"},
				{"slot", 0},
				{"attached_to", "nat"},
				{"adapter_type", "virtio-net"}
			},
			{
				{"name", "client_side"},
				{"slot", 1},
				{"attached_to", "internal"},
				{"network", "net1"},
				{"mac", "52:54:00:33:00:aa"}
			},
			{
				{"name", "server_side"},
				{"slot", 2},
				{"attached_to", "internal"},
				{"network", "net2"},
				{"mac", "52:54:00:33:00:bb"}
			},
			{
				{"name", "slave_side"},
				{"slot", 3},
				{"attached_to", "internal"},
				{"network", "net3"},
				{"mac", "52:54:00:33:00:cc"}
			}
		}},
		{"metadata", {
			{"login", "root"},
			{"password", "1111"}
		}},
		{"name", "controller"}
	};


static std::string snapsoht_cksum(const std::string& snapshot) {
	std::hash<std::string> h;
	return std::to_string(h(snapshot));
}

static void config_relevance_routine(const fs::path& testo_file, const nlohmann::json& original_config, bool is_relevant) {

	nlohmann::json new_config;

	Mock<Environment> mock_env;
	Mock<VmController> mock_vm;

	When(Method(mock_env, create_vm_controller)).Do([&](const nlohmann::json& a)->std::shared_ptr<VmController>{
		new_config = a;
		return std::shared_ptr<VmController>(&mock_vm.get(), [](VmController*){});
	});

	Fake(Method(mock_env, setup));
	Fake(Method(mock_env, cleanup));

	When(Method(mock_vm, keys)).AlwaysReturn({});
	When(Method(mock_vm, is_defined)).Return(true);
	When(Method(mock_vm, get_metadata).Using("vm_config")).Return(original_config.dump());
	When(Method(mock_vm, get_config)).Do([&]()->nlohmann::json {
		return new_config;
	});


	When(Method(mock_vm, has_snapshot).Using("dummy")).Return(true);
	When(Method(mock_vm, get_snapshot_cksum).Using("dummy")).Return(snapsoht_cksum("BLOCK"));
	Fake(Method(mock_vm, unplug_all_flash_drives));

	if (is_relevant) {
		When(Method(mock_vm, rollback).Using("dummy")).Return(0);

		//If install is invoked - that's an exception

		When(Method(mock_vm, install)).AlwaysThrow(std::runtime_error("Install was invoked instead of rollback"));
		{
			Interpreter runner(mock_env.get(), testo_file);
			runner.run();
		}

		REQUIRE_NOTHROW(Verify(Method(mock_vm, rollback)).Once());
	} else {
		When(Method(mock_vm, install)).Return(0);
		When(Method(mock_vm, name)).AlwaysReturn("controller");
		When(Method(mock_vm, make_snapshot)).AlwaysReturn(0);
		When(Method(mock_vm, set_snapshot_cksum)).AlwaysReturn(0);

		//If rollback is invoked - that's an exception
		When(Method(mock_vm, rollback)).AlwaysThrow(std::runtime_error("Rollback was invoked instead of install"));
		{
			Interpreter runner(mock_env.get(), testo_file);
			runner.run();
		}

		REQUIRE_NOTHROW(Verify(Method(mock_vm, install)).Once());
	}
}


TEST_CASE("config_original", "[rollback]") {
	config_relevance_routine("config_relevant/config_original.testo", default_original_config, true);
}

TEST_CASE("config_nics_mixed", "[rollback]") {
	config_relevance_routine("config_relevant/config_nics_mixed.testo", default_original_config, true);
}

TEST_CASE("config_metadata_mixed", "[rollback]") {
	config_relevance_routine("config_relevant/config_metadata_mixed.testo", default_original_config, true);
}

TEST_CASE("config_without_nics", "[rollback]") {
	nlohmann::json config = default_original_config;
	config.erase("nic");
	config_relevance_routine("config_relevant/config_without_nics.testo", config, true);
}

TEST_CASE("config_var_ref", "[rollback]") {
	setenv("ENV_VAR1", "/ubuntu-16.04.5", 1);
	setenv("ENV_VAR2", ".iso", 1);
	config_relevance_routine("config_relevant/config_var_ref.testo", default_original_config, true);
}

TEST_CASE("config_var_ref_changed", "[rollback]") {
	setenv("ENV_VAR1", "changed", 1);
	setenv("ENV_VAR2", ".iso", 1);
	config_relevance_routine("config_relevant/config_var_ref.testo", default_original_config, false);
}

//name of var is changed, but value is the same
TEST_CASE("config_var_ref_changed2", "[rollback]") {
	setenv("ENV_VAR5", "/ubuntu-16.04.5", 1);
	setenv("ENV_VAR6", ".iso", 1);
	config_relevance_routine("config_relevant/config_var_ref_name_changed.testo", default_original_config, true);
}

TEST_CASE("config_attr_changed", "[rollback]") {
	config_relevance_routine("config_irrelevant/config_attr_changed.testo", default_original_config, false);
}

TEST_CASE("config_attr_erased", "[rollback]") {
	config_relevance_routine("config_irrelevant/config_attr_erased.testo", default_original_config, false);
}

TEST_CASE("config_nic_erased", "[rollback]") {
	config_relevance_routine("config_irrelevant/config_nic_erased.testo", default_original_config, false);
}

TEST_CASE("config_nic_added", "[rollback]") {
	config_relevance_routine("config_irrelevant/config_nic_added.testo", default_original_config, false);
}

TEST_CASE("config_nic_renamed", "[rollback]") {
	config_relevance_routine("config_irrelevant/config_nic_renamed.testo", default_original_config, false);
}

TEST_CASE("config_nic_attr_changed", "[rollback]") {
	config_relevance_routine("config_irrelevant/config_nic_attr_changed.testo", default_original_config, false);
}
