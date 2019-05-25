
#include <iostream>
#include <chrono>
#include "api.hpp"
#include "virtual_box_client.hpp"

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

int main(int argc, char** argv) {
	try {
		vbox::API api;
		vbox::VirtualBoxClient client;
		auto virtual_box = client.virtual_box();

		auto start = std::chrono::high_resolution_clock::now();
		auto session = client.session();
		auto machine = virtual_box.find_machine("trololo2");
		machine.launch_vm_process(session, "headless").wait_and_throw_if_failed();
		session.unlock_machine();
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << time.count() << " seconds" << std::endl;

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
		return 1;
	}
}
