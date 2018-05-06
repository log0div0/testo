
#include <iostream>
#include <experimental/filesystem>
#include "vbox/api.hpp"
#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"
#include "vbox/lock.hpp"
#include "vbox/virtual_box_error_info.hpp"

namespace fs = std::experimental::filesystem;

void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << n << ". " << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << std::endl;
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << n << ". " << "[Unknown exception type]";
	}
}

std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}

int main(int argc, char* argv[]) {
	try {
		vbox::API api;
		vbox::VirtualBoxClient virtual_box_client;
		vbox::VirtualBox virtual_box = virtual_box_client.virtual_box();
		vbox::Session session = virtual_box_client.session();

		std::vector<vbox::Machine> machines = virtual_box.machines();
		for (auto& machine: machines) {
			if (machine.name() == "ubuntu_2") {
				machine.delete_config(machine.unregister(CleanupMode_DetachAllReturnHardDisksOnly)).wait_and_throw_if_failed();
			}
		}

		vbox::Machine machine = virtual_box.find_machine("ubuntu");
		std::cout << machine << std::endl;

		std::string settings_file = virtual_box.compose_machine_filename("ubuntu_2", "/", {}, {});
		std::cout << settings_file << std::endl;
		machine = virtual_box.create_machine(settings_file, "ubuntu_2", {"/"}, "ubuntu_64", {});
		machine.save_settings();
		virtual_box.register_machine(machine);
		{
			vbox::WriteLock lock(machine, session);
			vbox::Machine machine = session.machine();

			vbox::StorageController ide = machine.add_storage_controller("IDE", StorageBus_IDE);
			vbox::StorageController sata = machine.add_storage_controller("SATA", StorageBus_SATA);

			vbox::Medium dvd = virtual_box.open_medium("C:\\Users\\log0div0\\Downloads\\ubuntu-16.04.4-server-amd64.iso",
				DeviceType_DVD, AccessMode_ReadOnly, false);
			machine.attach_device(ide.name(), 1, 0, DeviceType_DVD, dvd);

			fs::path hard_disk_path = fs::path(machine.settings_file_path()).replace_extension("vdi");
			std::cout << hard_disk_path << std::endl;
			vbox::Medium hard_disk = virtual_box.create_medium("vdi", hard_disk_path.string(), AccessMode_ReadWrite, DeviceType_HardDisk);
			hard_disk.create_base_storage(8ull * 1024 * 1024 * 1024, {}).wait_and_throw_if_failed();
			machine.attach_device(sata.name(), 0, 0, DeviceType_HardDisk, hard_disk);
			machine.save_settings();
		}
		std::cout << machine << std::endl;
		return 0;
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
		return 1;
	}
}
