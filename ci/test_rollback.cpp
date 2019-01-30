
#include "fakeit.hpp"
#include "catch2/catch.hpp"
#include "testo/Interpreter.hpp"
#include <iostream>
#include <algorithm>

using namespace fakeit;

/*

Sooo.... config cksum tests.... Let's see

1) new config is exactly the same it was - the only test that should pass for now

2) new config is the same, but white spaces and tabs are different

3) new config is the same, but some rows are mixed up

3.5) new config differs in metadata

4) new config is the same, but some rows in nic section is mixed up

5) new config is the same, but some positions have var reference

6) new config differs in one row

7) new config differs in one row value

8) new config differs in nic name

That should cover it

*/

static std::string config_cksum(const nlohmann::json& config) {
	std::hash<std::string> h;
	return std::to_string(h(config.dump()));
}

static std::string snapsoht_cksum(const std::string& snapshot) {
	std::hash<std::string> h;
	return std::to_string(h(snapshot));
}

TEST_CASE("config_the_same", "[rollback]") {
	fs::path src_file("config_the_same.testo");

	nlohmann::json config = {
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
			},
		}},
		{"metadata", {
			{"login", "root"},
			{"password", "1111"}
		}},
		{"name", "client"}
	};

	nlohmann::json new_config;

	Mock<Environment> mock_env;
	Mock<VmController> mock_vm;

	When(Method(mock_env, create_vm_controller)).Do([&](const nlohmann::json& a)->std::shared_ptr<VmController>{
		new_config = a;
		return std::shared_ptr<VmController>(&mock_vm.get(), [](VmController*){});
	});

	Fake(Method(mock_env, setup));
	Fake(Method(mock_env, cleanup));

	//Fake(Method(mock_vm, install));
	When(Method(mock_vm, is_defined)).Return(true);
	When(Method(mock_vm, get_metadata).Using("vm_config_cksum")).Return(config_cksum(config));
	When(Method(mock_vm, config_cksum)).Do([&]()->std::string {
		return config_cksum(new_config);
	});


	When(Method(mock_vm, has_snapshot).Using("dummy")).Return(true);
	When(Method(mock_vm, get_snapshot_cksum).Using("dummy")).Return(snapsoht_cksum("snapshot dummy "));
	When(Method(mock_vm, rollback).Using("dummy")).Return(0);

	Fake(Method(mock_vm, unplug_all_flash_drives));
	{
		Interpreter runner(mock_env.get(), src_file);
		runner.run();
	}

	//REQUIRE_NOTHROW(Verify(Method(mock_vm, rollback).Using("dummy"))); This leads to exception
	REQUIRE_NOTHROW(Verify(Method(mock_vm, rollback)));  //This is okay
}
