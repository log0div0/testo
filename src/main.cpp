
#include <iostream>
#include "vbox/api.hpp"
#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"
#include "vbox/lock.hpp"

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
				std::vector<vbox::Medium> mediums = machine.unregister(CleanupMode_DetachAllReturnHardDisksOnly);
				vbox::Progress progress = machine.delete_config(std::move(mediums));
				progress.wait_for_completion();
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
			machine.add_storage_controller("IDE", StorageBus_IDE);
			machine.add_storage_controller("SATA", StorageBus_SATA);
			machine.save_settings();
		}
		std::cout << machine << std::endl;
		return 0;
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
		return 1;
	}
}
