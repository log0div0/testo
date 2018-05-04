
#include <iostream>
#include "vbox/api.hpp"
#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"

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
		std::vector<vbox::Machine> machines = virtual_box.machines();
		for (auto& machine: machines) {
			std::cout << machine.name() << std::endl;
		}
		std::vector<vbox::String> machine_groups = virtual_box.machine_groups();
		for (auto& machine_group: machine_groups) {
			std::cout << machine_group << std::endl;
		}
		return 0;
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
		return 1;
	}
}
