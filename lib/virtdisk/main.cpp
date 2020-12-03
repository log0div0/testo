
#include <iostream>
#include <chrono>
#include <thread>
#include "VirtualDisk.hpp"

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

void main() {
	try {
		auto start = std::chrono::high_resolution_clock::now();
		VirtualDisk virtual_disk("C:\\Users\\log0div0\\VirtualBox Flash Drives\\images\\pubkey_flash.vhd");
		// virtual_disk.attach();
		// virtual_disk.detach();
		std::cout << virtual_disk.isLoaded() << std::endl;
		std::cout << virtual_disk.physicalPath() << std::endl;
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << time.count() << " seconds" << std::endl;
	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}
