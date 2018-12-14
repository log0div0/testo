
#include <Interpreter.hpp>
#include <vbox/api.hpp>

#include <iostream>
#include <thread>
#include <chrono>

static void print_usage() {
	std::cout << "Usage: \n";
	std::cout << "testo <input file>\n";
}

int main(int argc, char** argv) {
	try {
		if (argc != 2) {
			print_usage();
		}

		/*nlohmann::json config_vm = {
			{"name", "client"},
			{"os_type", "ubuntu_64"},
			{"ram", 512},
			{"cpus", 2},
			{"disk_size", 4096},
			{"iso", "../iso/ubuntu-16.04.5-server-amd64.iso"}
		};*/

		/*nlohmann::json config = {
			{"name", "test-flash"},
			{"size", 256},
			{"fs", "ntfs"}
		};*/

		Interpreter runner(argv[1]);
		runner.run();
		
		/*std::shared_ptr<VboxVmController> vm(new VboxVmController(config_vm));
		std::cout << vm->is_additions_installed() << std::endl;
		auto res = vm->run("/bin/sh", {"-c", "set -e\n echo Hello world!\n sleep 3\n echo Narana"});
		std::cout << "Exit code: " << res << std::endl;*/
		//std::shared_ptr<FlashDriveController> fd(new FlashDriveController(config));
		//runner.global.vms.insert(std::make_pair("testo-ubuntu", vm));
		//runner.global.fds.insert(std::make_pair("test-flash", fd));
		//runner.run();
		//if (fd->create()) {
	//		return 0;
	//	}
	//	if (vm->plug_flash_drive(fd)) {
	//		throw;
	//	}
		//std::this_thread::sleep_for(std::chrono::seconds(50));
		//vm->unplug_flash_drive(fd);
		//vm->start();
		//vm->plug_flash_drive(fd);
		//vm->wait("Language", "5s");
		//fd->mount();
		//runner.run();

	/*	nlohmann::json config = {
			{"name", "testo-ubuntu"},
			{"os_type", "ubuntu_64"},
			{"ram", 512},
			{"cpus", 2},
			{"disk_size", 4096},
			{"iso", "../iso/ubuntu-16.04.5-server-amd64.iso"}
		};

		vbox::API vbox_api;

		std::cout << config.dump(4) << std::endl;
		std::shared_ptr<VboxVmController> vm(new VboxVmController(config));

		int res = vm->wait("Language", "5s");
		std::cout << res << std::endl;*/
		//std::this_thread::sleep_for(std::chrono::seconds(5));
		//vm->press({"LEFT"});
//		vm->press({"DOWN"});
//		vm->press({"DOWN"});
		//vm->press({"ENTER"});

		

		
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
		return -1;
	}

	

	return 0;
}
