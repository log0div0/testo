
#include "fakeit.hpp"
#include "catch2/catch.hpp"
#include "testo/Interpreter.hpp"
#include <iostream>

using namespace fakeit;

TEST_CASE("rollback") {
	fs::path src_file("test_rollback.testo");

	if (src_file.is_relative()) {
		src_file = fs::canonical(src_file);
	}

	Mock<Environment> mock_env;
	Mock<VmController> mock_vm;

	When(Method(mock_env, create_vm_controller)).Return(&mock_vm.get());
	Fake(Method(mock_env, setup));
	Fake(Method(mock_env, cleanup));

	Fake(Method(mock_vm, unplug_all_flash_drives));
	Interpreter runner(mock_env.get(), src_file);
	runner.run();

	Verify(Method(mock_vm, setup)).AtLeastOnce();
}
